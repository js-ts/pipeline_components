from typing import NamedTuple

def deploy_vertex_ai_model_to_endpoint(
    model_name: str,
    endpoint_name: str = None,
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = None,
    endpoint_display_name: str = None,
    deployed_model_display_name: str = None,
    project: str = None,
    location: str = "us-central1",
    timeout: float = None,
) -> NamedTuple("Outputs", [
    ("deployed_model_id", str)
]):
  """Deploys Google Cloud Vertex AI Model to a Google Cloud Vertex AI Endpoint.

  Args:
    model_name: Full resource name of a Google Cloud Vertex AI Model
    endpoint_name: Full name of Google Cloud Vertex Endpoint. A new
      endpoint is created if the name is not passed.
    machine_type: The type of the machine. See the [list of machine types
      supported for
      prediction](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types).
      Defaults to "n1-standard-2"
    min_replica_count: The minimum number of replicas the DeployedModel
      will be always deployed on. If traffic against it increases, it may
      dynamically be deployed onto more replicas up to max_replica_count, and as
      traffic decreases, some of these extra replicas may be freed. If the
      requested value is too large, the deployment will error. Defaults to 1.
    max_replica_count: The maximum number of replicas this DeployedModel
      may be deployed on when the traffic against it increases. If the requested
      value is too large, the deployment will error, but if deployment succeeds
      then the ability to scale the model to that many replicas is guaranteed
      (barring service outages). If traffic against the DeployedModel increases
      beyond what its replicas at maximum may handle, a portion of the traffic
      will be dropped. If this value is not provided, a no upper bound for
      scaling under heavy traffic will be assume, though Vertex AI may be unable
      to scale beyond certain replica number. Defaults to `min_replica_count`
    endpoint_display_name: The display name of the Endpoint. The name can
      be up to 128 characters long and can be consist of any UTF-8 characters.
      Defaults to the lowercased model ID.
    deployed_model_display_name: The display name of the DeployedModel. If
      not provided upon creation, the Model's display_name is used.
    project: The Google Cloud project ID. Defaults to the default project.
    location: The Google Cloud region. Defaults to "us-central1"
    timeout: Model deployment timeout
  """
  import logging
  import google
  import google.auth
  from google.cloud.aiplatform import gapic

  _logger = logging.getLogger(__name__)

  # Create an endpoint
  # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/create_endpoint_sample.py
  _, default_project = google.auth.default()
  if not project:
    project = default_project
  model_id = model_name.split("/")[-1]

  client_options = {
      "api_endpoint": f"{location}-aiplatform.googleapis.com",
  }
  endpoint_client = gapic.EndpointServiceClient(client_options=client_options)
  if not endpoint_name:
    if not endpoint_display_name:
      endpoint_display_name = model_id
    _logger.info("Creating new Endpoint: %s", endpoint_display_name)
    endpoint_to_create = {
        "display_name": endpoint_display_name,
    }
    endpoint = endpoint_client.create_endpoint(
        parent=f"projects/{project}/locations/{location}",
        endpoint=endpoint_to_create,
    ).result(timeout=timeout)
    endpoint_name = endpoint.name
  # projects/<project_id>/locations/<location>/endpoints/<endpoint_id>
  _logger.info("Endpoint name: %s", endpoint_name)

  # Deploy the model
  # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/deploy_model_custom_trained_model_sample.py
  model_to_deploy = {
      "model": model_name,
      "display_name": deployed_model_display_name,
      "dedicated_resources": {
          "min_replica_count": min_replica_count,
          "max_replica_count": max_replica_count,
          "machine_spec": {
              "machine_type": machine_type,
          },
      },
  }
  traffic_split = {"0": 100}
  _logger.info(
      "Deploying model %s to endpoint: %s", model_name, endpoint_display_name)
  deploy_model_operation = endpoint_client.deploy_model(
      endpoint=endpoint_name,
      deployed_model=model_to_deploy,
      traffic_split=traffic_split,
  )
  deployed_model = deploy_model_operation.result().deployed_model
  return deployed_model.id


if __name__ == "__main__":
  from kfp.components import create_component_from_func

  deploy_vertex_ai_model_to_endpoint_op = create_component_from_func(
    func=deploy_vertex_ai_model_to_endpoint,
    output_component_file="component.yaml",
    packages_to_install=[
      "google-cloud-aiplatform==1.3.0",
    ],
  )
