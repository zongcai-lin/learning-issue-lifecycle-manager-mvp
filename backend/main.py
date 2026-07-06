from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.analyzer import analyze_learning_conversation


app = FastAPI()

# 解决跨域问题（5000端口访问8000端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    conversation_text: str


@app.get("/")
def root():
    return {"message": "Learning Issue MVP backend is running"}


# @app.post("/analyze")
# def analyze_conversation(request: AnalyzeRequest):
#     result = analyze_learning_conversation(request.conversation_text)
#     return result

@app.post("/analyze")
def analyze_conversation(request: AnalyzeRequest):
    return {
        "goal_threads": [
            {
                "goal_id": "G1",
                "main_goal": "实现前端 Analyze 按钮成功请求 FastAPI 后端 /analyze",
                "status": "resolved",
                "summary": "这一条主线围绕后端接口测试、CORS 报错和前后端联通展开。",
                "issues": [
                    {
                        "title": "测试 /analyze 后端接口",
                        "type": "sub_issue",
                        "status": "resolved",
                        "trigger": "用户不知道如何在 FastAPI /docs 里测试接口",
                        "summary": "用户通过 /docs 页面测试 /analyze，确认后端可以返回假数据。",
                        "final_understanding": "如果 /docs 测试成功，说明后端接口本身没问题，前端失败可能来自 fetch、CORS 或页面代码。",
                        "review_note": "遇到前端按钮没反应时，先单独测试后端接口。"
                    },
                    {
                        "title": "解决 CORS 报错",
                        "type": "blocker",
                        "status": "resolved",
                        "trigger": "前端 5500 请求后端 8000 被浏览器拦截",
                        "summary": "用户理解了 5500 和 8000 是不同 origin，需要在 FastAPI 加 CORSMiddleware。",
                        "final_understanding": "CORS 是浏览器安全机制，后端需要明确允许前端来源访问。",
                        "review_note": "本地前后端不同端口时，很容易遇到 CORS。"
                    }
                ]
            },
            {
                "goal_id": "G2",
                "main_goal": "把 raw JSON 渲染成可读的 Issue List 页面",
                "status": "resolved",
                "summary": "这一条主线围绕 Step 5 页面渲染和必要 JavaScript 理解展开。",
                "issues": [
                    {
                        "title": "理解 result div 的作用",
                        "type": "sub_issue",
                        "status": "resolved",
                        "trigger": "用户不理解为什么页面里有空的 div id='result'",
                        "summary": "用户理解了 result div 是分析结果显示区域，不应该放进按钮内部。",
                        "final_understanding": "按钮负责触发动作，result div 负责显示结果。",
                        "review_note": "HTML 页面里常见结构是输入区、按钮、结果显示区分离。"
                    },
                    {
                        "title": "理解 fetch / JSON.stringify / response.json",
                        "type": "sub_issue",
                        "status": "resolved",
                        "trigger": "用户不理解前端如何请求后端并处理返回 JSON",
                        "summary": "用户理解了 fetch 负责请求后端，JSON.stringify 把前端对象转成 JSON 字符串，response.json 把后端 JSON 转成 JS 对象。",
                        "final_understanding": "前端和后端之间通过 JSON 格式交换数据。",
                        "review_note": "fetch 发送请求，JSON.stringify 发数据，response.json 读返回。"
                    }
                ]
            }
        ]
    }