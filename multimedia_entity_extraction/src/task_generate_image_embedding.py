
import os
from typing import Dict

from clearml import Task, Dataset

from config.config import cfg

Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("torch")


def download_datasets(cfg: Dict,) -> None:
    TEMP_PATH = '/tmp'
    for dataset in cfg['data']['clearml_datasets']:
        dataset_id = cfg['data']['clearml_datasets'][dataset]['id']
        clearml_dataset = Dataset.get(dataset_id=dataset_id)
        local_root_path = clearml_dataset.get_mutable_local_copy(
            os.path.join(TEMP_PATH, dataset_id))
        dataset_path = cfg['data']['clearml_datasets'][dataset]['path']
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        for f in os.listdir(local_root_path):
            full_f = os.path.join(local_root_path, f)
            new_f = os.path.join(dataset_path, f)
            os.rename(full_f, new_f)


def download_models(cfg: Dict) -> None:

    for model in cfg['training']['input_models']:
        model_id = cfg['training']['input_models'][model]['id']
        model_task = Task.get_task(task_id=model_id)
        last_snapshot = model_task.models['output'][-1]
        local_weights_path = last_snapshot.get_local_copy(
            os.path.join('/models', model))
        cfg['training']['input_models'][model]['path'] = local_weights_path


if __name__ == '__main__':
    PROJECT_NAME = 'datasets/multimodal'
    TASK_NAME = 'visual_embedding_generation'
    OUTPUT_URL = 's3://experiment-logging/multimodal'
    DATASET_NAME = 'visual_embeddings_flickr'
    DATASET_ROOT = '/data/embeddings/flickr_mscoco_visualgenome'

    # task = Task.init(project_name=PROJECT_NAME,
    #                  task_name=TASK_NAME, output_uri=OUTPUT_URL)
    # task.set_base_docker(
    #     docker_image=cfg['clearml']['base_image'],
    # )

    # task.connect(cfg)
    # task.execute_remotely(
    # queue_name = cfg['clearml']['queue'], exit_process = True)

    from data.image_tensor.visual_embedding_generator import EmbeddingGenerator

    # download_models(cfg)
    # download_datasets(cfg)
    task = None
    eg = EmbeddingGenerator(cfg, task)
    eg.run(DATASET_ROOT)

    # clearml_dataset = Dataset.create(
    #     dataset_project=PROJECT_NAME, dataset_name=DATASET_NAME)

    # clearml_dataset.add_files(DATASET_ROOT)
    # clearml_dataset.upload(show_progress=True,
    #                        verbose=True, output_url=OUTPUT_URL)
    # clearml_dataset.finalize()

    # exp.create_torchscript_model('class_model_v2.ckpt')
