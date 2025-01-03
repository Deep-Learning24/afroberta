import os
import shutil
from absl import app
from absl import flags

from src.trainer import TrainingManager
from src.inference import InferenceManager
from src.utils import load_config
import torch
import torch.distributed as dist

EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "config.yml"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "experiment_name",
    "",
    "Experiment name: experiment outputs will be saved in a created experiment name directory",
)
flags.DEFINE_string("config_path", "config.yml", "Config file path")
flags.DEFINE_integer(
    "world_size", 1, "Total number of GPUs to use for distributed training"
)
flags.DEFINE_integer(
    "rank", 0, "Rank of the current process for distributed training"
)
flags.DEFINE_string(
    "master_addr", "localhost", "Address of the master node for distributed training"
)
flags.DEFINE_integer(
    "master_port", 12393, "Port of the master node for distributed training"
)

def main(argv):
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Set up distributed training environment
    os.environ["RANK"] = str(FLAGS.rank)
    os.environ["WORLD_SIZE"] = str(FLAGS.world_size)
    os.environ["MASTER_ADDR"] = FLAGS.master_addr
    os.environ["MASTER_PORT"] = str(FLAGS.master_port)

    if FLAGS.world_size > 1:
        dist.init_process_group(backend="nccl")

    # Load config
    config = load_config(FLAGS.config_path)

    # Set up experiment directory
    experiment_path = os.path.join(EXPERIMENT_PATH, FLAGS.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)
    shutil.copy2(FLAGS.config_path, experiment_config_path)

    # Initialize and start training
    trainer = TrainingManager(config, experiment_path)
    # trainer.train()
    

    # if FLAGS.world_size > 1:
    #     dist.destroy_process_group()


    # Inference
    # Define paths and configuration
    CONFIG_PATH = "mlm_configs/afriberta_base.yml"
    MODEL_PATH = "experiments/afriberta_base"
    TEST_FILE_PATH = "data/test/test.kin"

    # Initialize InferenceManager
    inference_manager = InferenceManager(CONFIG_PATH, trainer.model,trainer.tokenizer, TEST_FILE_PATH)

    # Load test data
    test_sentences = inference_manager.load_test_data()

    # Run evaluation
    results, bleu_score,avg_rouge_scores = inference_manager.evaluate(test_sentences)
    
    # Print results
    for i,result in enumerate(results):
        print(f": Result {i+1}: {result}")
    print(f"Blue score is: {bleu_score}")
    print(f"avg_rouge_scores: {avg_rouge_scores}")
        

if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_name")
    app.run(main)
