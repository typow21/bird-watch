from locust import HttpUser, TaskSet, task, events, between
import time

class MyTaskSync(TaskSet):
    @task
    def my_task(self):
        self.client.get("/sync")

class MyTaskAsyncBG(TaskSet):
    @task
    def my_task(self):
        self.client.get("/")

class MyUser(HttpUser):
    tasks = [MyTaskAsyncBG]
    min_wait = 0
    max_wait = 0
    wait_time = between(min_wait, max_wait)

# def on_hatch_complete(user_count, **kwargs):
#     start_time = time.time()
#     def quit(reason=None):
#         end_time = time.time()
#         total_time = end_time - start_time
#         if total_time >= 180:
#             events.quitting.fire(reason="Complete the test for 60 seconds")
#     events.quitting += quit
#     events.hatch_complete += on_hatch_complete
