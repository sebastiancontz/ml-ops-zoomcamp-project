from prefect.deployments import Deployment

from real_estate_model.flows.model_training import model_training
from real_estate_model.flows.model_batch_prediction import batch_prediction

def deploy(flow):
    deployment = Deployment.build_from_flow(
        flow=flow,
        name=flow.__name__
    )
    deployment.apply()

if __name__ == "__main__":
    flows = [model_training, batch_prediction]
    for flow in flows:
        deploy(flow)