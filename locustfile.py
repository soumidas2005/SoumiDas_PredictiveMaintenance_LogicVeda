from locust import HttpUser, task, between

class DashboardUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_dashboard(self):
        self.client.get("/")