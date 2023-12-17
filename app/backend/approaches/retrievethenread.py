import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines

# Cognitive SearchとOpenAIのAPIを直接使用した、シンプルなretrieve-then-readの実装です。これは、最初に
# 検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する 
# (answer)をそのプロンプトで表示します。

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):

    template = \
"あなたは自動車に関する質問をサポートする教師アシスタントです。" + \
"質問者が「私」で質問しても、「あなた」を使って質問者を指すようにする。" + \
"次の質問に、以下の出典で提供されたデータのみを使用して答えてください。" + \
"各出典元には、名前の後にコロンと実際の情報があり、回答で使用する各事実には必ず出典名を記載します。" + \
"以下の出典の中から答えられない場合は、「わかりません」と答えてください。" + \
"""

###
Question: 'EVとはなんですか?'

Sources:
info1.txt: 「EV」は「Electric Vehicle」の略で、電気自動車のことです
info2.txt: モーターを動力として走行し、二酸化炭素を排出しないため、環境負荷が小さい

Answer:
「EV」は「Electric Vehicle」の略で、電気自動車のこと[info1.txt]です。モーターを動力として走行し、二酸化炭素を排出しないため、環境負荷が小さい[info2.txt]という特徴があります

###
Question: '{q}'?

Sources:
{retrieved}

Answer:
"""

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:

        #prompt = (overrides.get("prompt_template") or self.template).format(q=q, retrieved=content)
        prompt = "You are a translation assistant. Please answer by translating the characters you entered into English. Please write your answers in English only."
        prompt = prompt + " Question: '" + q + "'"
        completion = openai.Completion.create(
            engine=self.openai_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.3, 
            max_tokens=2048, 
            n=1, 
            stop=["\n"])

        return {"data_points": "", "answer": completion.choices[0].text, "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
