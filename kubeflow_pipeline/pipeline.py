import os
import kfp
from kfp import dsl


@dsl.pipeline(name="mnist pipeline", description="mnist pipeline")
def mnist_pipeline():

    data_0 = dsl.ContainerOp(
        name="load data pipeline",
        image="ssuwani/mnist_0_data",
    ).set_display_name("collect data")

    preprocess_data_1 = (
        dsl.ContainerOp(
            name="preprocess data pipeline",
            image="ssuwani/mnist_1_preprocess_data",
        )
        .set_display_name("preprocess data")
        .after(data_0)
    )

    train_model_2 = (
        dsl.ContainerOp(
            name="train model",
            image="ssuwani/mnist_2_train_model",
        )
        .set_display_name("train model")
        .after(preprocess_data_1)
    )


if __name__ == "__main__":
    host = "https://17c0fa73382ff154-dot-asia-east1.pipelines.googleusercontent.com/"

    pipeline_name = "mnist"
    pipeline_package_path = "pipeline.zip"

    client = kfp.Client(host=host)
    kfp.compiler.Compiler().compile(mnist_pipeline, pipeline_package_path)

    client.create_run_from_pipeline_func(
        mnist_pipeline,
        arguments={},
    )
