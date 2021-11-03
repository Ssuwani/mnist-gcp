import kfp
from kfp.v2 import dsl
from kfp import components

@dsl.pipeline(
    name="mnist-pipeline",
    description="mnist pipeline tutorial",
    pipeline_root="gs://suwan/mnist-20211104"
)
def pipeline():
    load_data = components.load_component_from_file("0_load_data.yaml")
    load_data_task = load_data()
    
if __name__ == "__main__":
    
    client = kfp.Client("https://22767e8e71dc72a3-dot-us-central1.pipelines.googleusercontent.com/")
    
    client.create_run_from_pipeline_func(
        pipeline,
        arguments={},
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
    )    
