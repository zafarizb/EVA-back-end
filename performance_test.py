from locust import HttpUser, TaskSet, task

class UserTasks(TaskSet):
    # 列出需要测试的请求形式
    @task(1)
    def index(self):
        self.client.get("/main")
    @task(1)
    def image(self):
        self.client.get("/image/?userid=1")
    @task(1)
    def image_detail(self):
        self.client.get("/image/detail?url=%2FPage%2Fruntest2&userid=1")
    @task(1)
    def video(self):
        self.client.get("/video/?userid=1")
    @task(1)
    def video_detail(self):
        self.client.get("/video/detail?url=%2FVideo%2Fvideotest1&userid=1")
    @task(1)
    def file(self):
        self.client.get("/file/?userid=1")

class WebsiteUser(HttpUser):
    host = "http://127.0.0.1:8000"
    tasks = [UserTasks]
    min_wait = 3000  # 模拟负载的任务之间执行的最小等待时间，单位为毫秒
    max_wait = 6000  # 模拟负载的任务之间执行的最大等待时间，单位为毫秒


