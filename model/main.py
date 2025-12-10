import pandas as pd
import numpy as np
import json
import pprint
import string
import re
import time
from datetime import timedelta, datetime
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoFeatureExtractor, AutoModel
import multiprocessing
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from tqdm import tqdm
from newsplease import NewsPlease
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
# import preprocessor as p
# p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from google import genai


def gemini(api_key, prompt):
    max_retries = 2
    curr_tries = 1
    while curr_tries <= max_retries:
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            
            return response.text
        except Exception as e:
            print('\t'+str(e))
            curr_tries += 1
            time.sleep(5)
            continue
    return ''


def query_generation(news, llm_key):
    # Prepare news data
    def news_prep(news_category, news_text):
        return f'news category: {news_category}, content: {news_text}'

    news_content = news_prep(news['CATEGORY'], news['TEXT'])

    # Load the prompt template
    with open('data/prompt_query_generation_unimodal.txt', 'r') as f:
        prompt_format = f.read()
    # Prepare the prompt
    prompt = prompt_format.replace('[NEWS_CONTENT]', news_content)
    # Generate queries with Gemini
    response = gemini(llm_key, prompt)

    return response


def query_search(api_key, search_engine_id, news, domain_priority):
    def google_programmable_search(api_key, search_engine_id, query):
        params = {'key': api_key, 'cx': search_engine_id, 'q': query}
        search = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        results = search.json()
        return results

    search_results = {}
    queries = news["queries"]
    # No queries, return {}
    if queries in ["NONE", "\"NONE\""]:
        return search_results
    # Otherwise, [query_1, query_2, ...]
    queries = queries.split('\n')
    queries = [q for q in queries if len(q)>0 and q[0] in ['1', '2', '3']]
    # print('---------query---------')
    for x, query_x in enumerate(queries):
        qx = ' '.join(query_x.split(' ')[1:])
        # print(qx)
        qx = qx.translate(str.maketrans('', '', string.punctuation))
        save_file = 'data/web_search_text/'+qx+'_'+domain_priority+'.json'
        # If the search results have not been saved, do web search. Otherwise, load the search results
        if not os.path.exists(save_file):
            with open(save_file, 'w') as f:
                json.dump({}, f, indent=4)
        with open(save_file, 'r') as f:
            google_programmable_search_outputs = json.load(f)
        if 'items' in google_programmable_search_outputs.keys():
            search_results[qx] = google_programmable_search_outputs
        else:
            output = google_programmable_search(api_key, search_engine_id, qx)
            print("\t(sent 1 request to google programmable search)")
            if 'items' in output.keys():
                search_results[qx] = output
            google_programmable_search_outputs = output
            with open(save_file, 'w') as f:
                json.dump(google_programmable_search_outputs, f, indent=4)
    return search_results


def filter_search_results_by_domain(search_results, domain_priority, page_num=-1):
    with open('data/publisher_priority_' + domain_priority.lower() + '.txt', 'r') as f:
        selected_domains = f.read().split('\n')

    if page_num == -1:
        sr = search_results
    else:
        sr = search_results['page_' + str(page_num+1)]

    filtered_search_links = set()
    for results in sr.values():
        if 'image_results' not in results.keys():
            continue
        for res in results['image_results']:
            domain = res['link'].split('/')[2]
            domain = domain.replace('www.', '')
            if domain in selected_domains:
                filtered_search_links.add(res['link'])

    return filtered_search_links


def filter_search_results_by_query_sim(search_results, top_k):
    filtered_search_links = set()
    for results in search_results.values():
        for i in range(min(top_k, len(results['items']))):
            filtered_search_links.add(results['items'][i]['link'])
    return list(filtered_search_links)


def article_crawler(article_urls, driver):

    searched_articles = []

    articles_json = 'data/web_page_content.json'
    if not os.path.exists(articles_json):
        with open(articles_json, 'w') as f:
            json.dump({}, f)
    with open(articles_json, 'r') as f:
        saved_articles = json.load(f)

    for _, article_url in tqdm(enumerate(article_urls)):
        # Extract the content
        if article_url in saved_articles.keys():
            article = saved_articles[article_url]
        else:
            try:
                driver.get(article_url)
            except Exception as e:
                print('\t(fail to get ' + article_url + ')')
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            article = NewsPlease.from_html(soup.prettify(), article_url).get_dict()

            saved_articles[article_url] = article
            with open(articles_json, 'w') as f:
                json.dump(saved_articles, f, sort_keys=True, default=str, indent=4)

        # Process the content
        if article['maintext'] and len(article['maintext'].split()) >= 100 and article['date_publish'] and article['language'] == 'en':
            # Remove the duplicate content
            flag = 0
            for searched_article in searched_articles:
                curr_article_time = datetime.strptime(str(article['date_publish']), '%Y-%m-%d %H:%M:%S')
                ref_article_time = datetime.strptime(str(searched_article['date_publish']), '%Y-%m-%d %H:%M:%S')
                if article['maintext'] in searched_article['maintext'] and (curr_article_time - ref_article_time).total_seconds() >= 0:
                    flag = 1
                    break
                elif searched_article["maintext"] in article["maintext"] and (ref_article_time - curr_article_time).total_seconds() >= 0:
                    searched_articles.remove(searched_article)
                    searched_articles.append(article)
                    flag = 1
                    break
            if flag == 0:
                searched_articles.append(article)

    return searched_articles


def compute_unimodal_sims(news, article):
    def get_text_sim(model, text_1, text_2):
        embeds_1 = model.encode(text_1, convert_to_tensor=True).cpu()
        embeds_2 = model.encode(text_2, convert_to_tensor=True).cpu()
        return util.dot_score(embeds_1, embeds_2).numpy()[0][0]

    articleUrl_textSim = {article['url']: float(-1)}

    text_sim = get_text_sim(SentenceTransformer('msmarco-distilbert-base-tas-b'),
                            news['TEXT'], article['maintext'])
    articleUrl_textSim[article['url']] = text_sim

    return articleUrl_textSim


def filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim, searched_articles, max_num=3, threshold=90):
    filtered_searched_links = set()

    article_urls = [article_url for articleUrl_textSim in list_of_articleUrl_textSim for article_url in
                    articleUrl_textSim.keys()]
    text_sims = [text_sim for articleUrl_textSim in list_of_articleUrl_textSim for text_sim in
                 articleUrl_textSim.values()]

    textSims_articleUrls = list(zip(text_sims, article_urls))
    textSims_articleUrls.sort(reverse=True)

    article_urls = [article_url for _, article_url in textSims_articleUrls]
    text_sims = [text_sim for text_sim, _ in textSims_articleUrls]

    x = min(len(text_sims), max_num)
    for idx in range(x):
        if text_sims[idx] >= threshold:
            filtered_searched_links.add(article_urls[idx])

    filtered_searched_articles = []
    for article in searched_articles:
        if article['url'] in filtered_searched_links:
            filtered_searched_articles.append(article)

    return filtered_searched_articles


def evidence_extraction(news, article, llm_key):
    output = ([], [])

    news_category = news['CATEGORY']
    news_text = news['TEXT']

    article_content = article['maintext']
    article_time = str(article['date_publish'])
    article_url = article['url']

    prompt = ''
    # Use the first max_chars characters
    article_content = article_content[:20000]

    with open('data/prompt_evidence_extraction.txt', 'r') as f:
        prompt_format = f.read()
    prompt = prompt_format.replace('[ARTICLE_CONTENT]', article_content)
    prompt = prompt.replace('[NEWS_CONTENT]', f'news category: {news_category}, content: {news_text}')

    evidence = gemini(llm_key, prompt)
    print("\t(sent 1 request to llm)")

    if 'explicit refutation' in evidence.lower() and 'implicit refutation' in evidence.lower():
        evidences = evidence.split('\n')
        for i, evidence in enumerate(evidences):
            if 'explicit refutation' in evidence.lower():
                er_loc = i
            elif 'implicit refutation' in evidence.lower():
                ce_loc = i
        refute_evidence, context_evidence = evidences[er_loc+1:ce_loc], evidences[ce_loc+1:]

        refute_evidence_2 = []
        for i, s in enumerate(refute_evidence):
            if 'NONE' in s or len(s)==0 or (len(s)>0 and s[0]!='-'):
                continue
            if article_time != '':
                subs = ' Reference: ' + article_url + ' published on ' + article_time
            else:
                subs = ' Reference: ' + article_url
            if len(s)>1 and s[2] == "\"" and s[-1] == "\"":
                refute_evidence_2.append('- ' + s[3:-1] + subs)
            else:
                refute_evidence_2.append(s + subs)
        # print('---------refute_evidence---------')
        # print(refute_evidence_2)

        context_evidence_2 = []
        for i, s in enumerate(context_evidence):
            if 'NONE' in s or len(s)==0 or (len(s)>0 and s[0]!='-'):
                continue
            if article_time != '':
                subs = ' Reference: ' + article_url + ' published on ' + article_time
            else:
                subs = ' Reference: ' + article_url
            if len(s)>1 and s[2] == "\"" and s[-1] == "\"":
                context_evidence_2.append('- ' + s[3:-1] + subs)
            else:
                context_evidence_2.append(s + subs)
        # print('---------context_evidence---------')
        # print(context_evidence_2)

        output = (refute_evidence_2, context_evidence_2)

    return output


def correction_generation(news, list_of_evidences, list_of_refute_evidences, llm_key):
    if len(list_of_refute_evidences) == 0:
        with open('data/prompt_correction_generation_without_retrieval.txt', 'r') as f:
            prompt = f.read()
    else:
        with open('data/prompt_correction_generation_with_retrieval.txt', 'r') as f:
            prompt = f.read()
        prompt = prompt.replace('[RELEVANT_FACTS]', '\n'.join(list_of_evidences))

    news_category = news['CATEGORY']
    news_text = news['TEXT']

    prompt.replace('[NEWS_CONTENT]', f'news category: {news_category}, content: {news_text}')

    return gemini(llm_key, prompt)


def get_article_publish_date(article):
    return datetime.strptime(str(article['date_publish']), "%Y-%m-%d %H:%M:%S")


if __name__ == '__main__':
    TEST_NEWS_ID = '1'
    data_file_id = '1QSZ8WRe2747lzueN5vkDP3ohbJavUNCH'
    download_url = f'https://drive.google.com/uc?export=download&id={data_file_id}'

    # Read the file
    data = pd.read_csv(download_url, dtype=str, encoding='latin-1')
    data = data.fillna('')

    instance = data[data['ID'] == TEST_NEWS_ID].to_dict('records')[0]

    # Load API keys
    with open('data/api_keys.json', 'r') as f:
        api_keys = json.load(f)
    llm_key = api_keys['gemini']
    # llm_key = api_keys['HuggingFace']
    serpapi_key = api_keys['SerpAPI']
    gsearch_key = api_keys['GoogleSearch']['Key']

    # Open a browser for article crawling
    options = Options()
    driver = webdriver.Firefox(options=options)
    driver.implicitly_wait(5)
    driver.set_page_load_timeout(10)

    # Set the number of processes = 5
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=5)

    print('Start correcting news')

    print('Generate queries from misinformation...')
    instance['queries'] = query_generation(instance, llm_key)

    refute_evidences, context_evidences = set(), set()
    for domain_priority in ['High', 'Medium', 'Low']:
        print('Search web pages with ' + domain_priority + ' priority...')
        search_results = query_search(gsearch_key, api_keys['GoogleSearch'][domain_priority+'-Priority'], instance, domain_priority)

        print('Start selecting retrieved web pages...')

        # Filtered based on their similarities with the query
        filtered_search_links = filter_search_results_by_query_sim(search_results, 3)

        print('Extract retrieved web page content...')
        searched_articles = article_crawler(filtered_search_links, driver)
        if searched_articles == []:
            print('\t(no article found)')
            continue

        # Filtered based on their similarities with the misinfo content
        print('Compute the similarity between web page and misinformation content...')
        params = [(instance, article) for article in searched_articles]
        list_of_articleUrl_textSim = pool.starmap(compute_unimodal_sims, params)
        final_searched_articles = filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim, searched_articles)
        if final_searched_articles == []:
            print('\t(no article qualified based on similarity)')
            continue

        print('Extract evidence from selected web pages...')
        params = [(instance, article, llm_key) for article in final_searched_articles]
        evidences = pool.starmap(evidence_extraction, params)

        for refute_evidence, context_evidence in evidences:
            refute_evidences = refute_evidences.union(set(refute_evidence))
            context_evidences = context_evidences.union(set(context_evidence))

        if len(refute_evidences) >= 3:
            print('\t(stop with sufficient refutations)')
            break

    if len(refute_evidences) == 0:
        print('\nExtend web search due to no refutations...')

        refute_evidences, context_evidences = set(), set()
        for domain_priority in ['High', 'Medium', 'Low']:
            print('Search web pages with ' + domain_priority + ' priority...')
            search_results = query_search(gsearch_key, api_keys['GoogleSearch'][domain_priority + '-Priority'],
                                            instance, domain_priority)

            print('Start selecting retrieved web pages...')

            # Filtered based on their similarities with the query
            filtered_search_links = filter_search_results_by_query_sim(search_results, 10)

            print('Extract retrieved web page content...')
            searched_articles = article_crawler(filtered_search_links, driver)
            if searched_articles == []:
                print('\t(no article found)')
                continue

            # Filtered based on their similarities with the misinfo content
            print('Compute the similarity between web page and misinformation content...')
            params = [(instance, article) for article in searched_articles]
            list_of_articleUrl_textSim = pool.starmap(compute_unimodal_sims, params)
            final_searched_articles = filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim,
                                                                            searched_articles)
            if final_searched_articles == []:
                print('\t(no article qualified based on similarity)')
                continue

            print('Extract evidence from selected web pages...')
            params = [(instance, article, llm_key) for article in final_searched_articles]
            evidences = pool.starmap(evidence_extraction, params)

            for refute_evidence, context_evidence in evidences:
                refute_evidences = refute_evidences.union(set(refute_evidence))
                context_evidences = context_evidences.union(set(context_evidence))

            if len(refute_evidences) >= 3:
                print('\t(stop with sufficient refutations)')
                break

    refute_evidences = list(refute_evidences)
    context_evidences = list(context_evidences)

    instance['refute_evidence'] = refute_evidences
    instance['context_evidence'] = context_evidences

    evidences = refute_evidences + context_evidences

    print('Generate correction with ' + str(len(refute_evidences)) + ' refutations...')
    instance['correction'] = correction_generation(instance, evidences, refute_evidences, llm_key)

    with open('data/output/' + instance["ID"] + ".json", 'w') as f:
        json.dump(instance, f, indent=4, sort_keys=True)

    print()
    print(instance['ID'])
    print(instance['correction'])

    driver.quit()
