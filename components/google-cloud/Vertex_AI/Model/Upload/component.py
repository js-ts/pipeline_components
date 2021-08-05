from typing import NamedTuple


def create_model_in_Google_Cloud_Vertex_AI(
    serving_container_image_uri: str,
    artifact_uri: str = None,
    display_name: str = "Model",
    project: str = None,
    location: str = "us-central1",
) -> NamedTuple("Outputs", [
    ("model_name", str),
]):
  from google.cloud import aiplatform

  aiplatform.init(
      project=project,
      location=location,
  )
  model = aiplatform.Model.upload(
      display_name=display_name or "Model",
      serving_container_image_uri=serving_container_image_uri,
      artifact_uri=artifact_uri,
  )
  return (model.name,)


if __name__ == "__main__":
  from kfp.components import create_component_from_func

  create_model_in_Google_Cloud_Vertex_AI_op = create_component_from_func(
    func=create_model_in_Google_Cloud_Vertex_AI,
    output_component_file="component.yaml",
    packages_to_install=[
      "google-cloud-aiplatform==1.3.0",
    ],
  )
