from prefect.deployments import Deployment

from real_estate_model.flows.model_training import model_training

def deploy():
    deployment = Deployment.build_from_flow(
        flow=model_training,
        name="model_training"
    )
    deployment.apply()

if __name__ == "__main__":
    deploy()