import os

# from chat.chat_args_ import parse_args_
from chat.topk import get_top_k  # , get_top_k_faiss

from chat.crawling.google import google_search
from chat.crawling.naver import naver_search
from chat.crawling.extractor import MainTextExtractor
from chat.run_add_query import run_add_query

import openai
import datetime

import yaml
import requests
import asyncio
import aiohttp

from readability import Document
from bs4 import BeautifulSoup

from kiwipiepy import Kiwi

class Args:
    def __init__(self):
        self.query = None
        self.use_google = 1
        self.google_n_pages = 1
        self.use_naver = 0
        self.naver_display = 20
        self.calculated_for = 'snippet'
        self.use_faiss = 0
        self.top_k = 3

args_ = Args()


def run_chat(query):
    with open("chat/API.yaml", "r") as yaml_conf:
        conf = yaml.safe_load(yaml_conf)
        API = conf["API"]

    args_.query = query
    kiwi = Kiwi()


    args_.query = query

    print(f"전처리 전 쿼리: {query}, 전처리 후 쿼리: {args_.query}")

    if API["google_search_engine_id"] is None or API["google_api_key"] is None:
        raise Exception("Insert your own Google Search API into args.py.")
    if API["naver_client_id"] is None or API["naver_client_secret"] is None:
        raise Exception("Insert your own NAVER Search API into args.py.")
    if args_.query is None:
        raise Exception("—query is required.")
    corpus_list = []
    links = []
    if args_.use_google:
        corpus_list, links = google_search(args_, corpus_list, links)
    if args_.use_naver:  # 현재 미사용 설정
        corpus_list, links = naver_search(args_, corpus_list, links)

    if not corpus_list:
        raise Exception("You must use at least one search engine.")

    ### 구글 상위 가져오기 (07/24)
    corpus_list, result_links = corpus_list[: args_.top_k], links[: args_.top_k]

    urls = ["http://115.85.181.95:30013/get_prediction/", "http://49.50.172.150:40001/get_prediction/", "http://49.50.160.171:30003/get_prediction/"]
    print(f"result_links: \n {result_links}")
    summaries = []

    async def req(link, url):
        doc = Document(requests.get(link).content)
        main_content = BeautifulSoup(doc.summary(), "lxml").text


        main_content = main_content.replace('\n', ' ')
        print(f"main_content: \n {main_content}")
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                url,
                params={
                    "input": main_content[:2000]
                    if len(main_content) > 2000
                    else main_content
                },
            )
            data = await response.json()
            print('############### COMPLETE ###############')
            return data["output"].split("### 요약:")[1].split("<|endoftext|>")[0]


    async def req_main(result_links, urls):
        tasks = [
            # 비동기로 실행할 작업을 생성하고, 해당 작업들을 테스크 리스트에 추가
            # req(link, url) 함수를 비동기적으로 실행하여 서버에 요청을 보내고 응답을 기다림
            asyncio.create_task(req(link, url))
            for link, url in zip(result_links, urls)
        ]
        # 생성한 task들을 asyncio.gather 함수로 실행하여 모든 작업들이 완료될 때까지 기다림
        # return_exceptions=True: 오류가 발생한 경우엗 결과를 기다리도록 함
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses

    summaries = asyncio.run(req_main(result_links, urls))

    # None인 응답 (오류가 발생한 경우)을 필터링하고, summaries 리스트에 추가합니다.
    summaries = [summary for summary in summaries if summary is not None]

    summaries_merge = ""
    try:
        print("TRY")
        for idx, summary in enumerate(summaries):
            summaries_merge += f"[{idx+1}] {summary} "
        print(summaries_merge)

        print(len(summaries_merge))

    except:
        print("EXCEPT")
        print(summaries)

    ###############################################################################################

    openai.api_key = API["openai_api_key"]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature = 0.5,
        top_p = 0.95,
        messages=[
            {
                "role": "user",
                "content": f"{summaries_merge}, This is a summary of the three articles. Please generate a response for the given {query}. According to the following rules 1. Use only the provided summary informations. 2. Keep it under 80 words. 3. Write in an unbiased and objective tone. 4. Pay attention to spelling and context 5. Provide answer in Korean",
            },
        ],
    )
    answer = completion["choices"][0]["message"]["content"]

    nexts = run_add_query(answer, query, summaries_merge)

    return answer, result_links, nexts
