import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines
import json
import requests

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.

# Cognitive SearchとOpenAIのAPIを直接使用した、シンプルな retrieve-then-read の実装です。これは、最初に
# 検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する (answer)をそのプロンプトで表示します。


def prompt_settings(mode) -> any:
    print(mode)
    entries_get_url = "https://dcj-tst.outsystemsenterprise.com/SYSNAVI_MAINTE/rest/RESTAPI1/RESTAPIMethod2?TemplateKey=" + mode
    entries_headers = {'content-type': 'application/json'}
    entries_response = requests.get(url=entries_get_url, headers=entries_headers)
    json_text = json.loads(entries_response.text)
    print(json_text)
    return json_text

class ChatReadRetrieveReadApproach_St(Approach):
    orgprompt_prefix = """<|im_start|>system
自動車に関する質問をサポートする教師アシスタントです。回答は簡潔にしてください。
以下の出典リストに記載されている事実のみを回答してください。以下の情報が十分でない場合は、「わからない」と答えてください。以下の出典を使用しない回答は作成しないでください。ユーザーに明確な質問をすることが助けになる場合は、質問してください。
各出典元には、名前の後にコロンと実際の情報があり、回答で使用する各事実には必ず出典名を記載してください。ソースを参照するには、四角いブラケットを使用します。例えば、[info1.txt]です。出典を組み合わせず、各出典を別々に記載すること。例えば、[info1.txt][info2.pdf] など。
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}
"""
    prompt_prefix = """<|im_start|>system
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}
"""

    follow_up_questions_prompt_content = """自動車について、ユーザーが次に尋ねそうな非常に簡潔なフォローアップ質問を3つ作成する。
    質問を参照するには、二重の角括弧を使用します（例：<<EVとはなにですか?>>）。
    すでに聞かれた質問を繰り返さないようにしましょう。
    質問のみを生成し、「次の質問」のような質問の前後にテキストを生成しない。"""

    query_prompt_template = """以下は、これまでの会話の履歴と、自動車に関するナレッジベースを検索して回答する必要がある、ユーザーからの新しい質問です。
    会話と新しい質問に基づいて、検索クエリを作成します。
    検索クエリには、引用元のファイル名や文書名（info.txtやdoc.pdfなど）を含めないでください。
    検索キーワードに[]または<<>>内のテキストを含めないでください。

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict, mode) -> any:
        json_text = prompt_settings(mode)
        self.prompt_prefix = json_text['prompt_prefix'] if json_text['prompt_prefix'].strip() != "" else self.prompt_prefix
        use_semantic_captions = True if json_text['semantic_captions'].strip() != "" else False
        top = json_text['top'] if json_text['top'].strip() != "" else 3
        exclude_category = json_text['exclude_category'] if json_text['exclude_category'].strip() != "" else None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text
        print(q)
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        smranker = True if json_text['semantic_captions'].strip() != "" else False
        if smranker:
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="ja-jp", 
                                          query_speller="none", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)
        follow_up_questions_prompt = json_text['suggest_followup_questions'] if json_text['suggest_followup_questions'].strip() != "" else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = json_text['prompt_template'] if json_text['prompt_template'].strip() != "" else None
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        print(len(prompt),prompt)
        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        temperature_text = json_text['temperature'] if json_text['temperature'].strip() != "" else '0.0'
        print("temperature_text:" + temperature_text)
        temperature_val = float(temperature_text)
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment, 
            prompt=prompt, 
            temperature=temperature_val, 
            max_tokens=2048, 
            n=1,
            stream=True,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>", "<|im_start|>"])
        return completion, {"data_points": results, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
