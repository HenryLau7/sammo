import click
import sammo
import orjson

from sammo.mutators import (
    BagOfMutators,
    APE,
    InduceInstructions,
    SyntaxTreeMutator,
    APO,
    Paraphrase,
    ChangeDataFormat,
    DecreaseInContextExamples,
)
from sammo.runners import OpenAIChat, Vllm, AzureGPT4
from sammo.throttler import AtMost

logger = sammo.setup_logger(log_prompts_to_file=True)

from sammo import search_op
from sammo.data import DataTable
from sammo.instructions import MetaPrompt, Paragraph, InputData, FewshotExamples
from sammo.components import Output
from sammo.dataformatters import PlainFormatter, JSONDataFormatter, XMLDataFormatter, QuestionAnswerFormatter, MultiLabelFormatter
from sammo.search import EnumerativeSearch, BeamSearch, Optimizer
from sammo.store import PersistentDict

import pathlib
from utils import accuracy, accuracy_GSM8K, accuracy_MATH

MAIN_FOLDER = sammo.utils.DEFAULT_SAVE_PATH
CONFIG_PATH = MAIN_FOLDER.parent.parent.parent / "config"
MODEL_CONFIGS = {
    "gpt-3.5": {
        "full_id": "gpt-3.5-turbo-16k-0613",
        "equivalence_class": "gpt-3.5-turbo-16k",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    },
    "gpt-4": {
        "full_id": "gpt-4-0613",
        "equivalence_class": "gpt-4-0613",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    },
    "llama-2": {
        "full_id": "meta-llama/Llama-2-70b-chat-hf",
        "equivalence_class": "meta-llama/Llama-2-70b-chat-hf",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": 4096,
    },
    "mixtral": {
        "full_id": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "equivalence_class": "dolphin-2.6-mixtral-8x7b",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": None,
    },
    "vllm": {
        "full_id": "vllm",
        "equivalence_class": "vllm",
        "credentials": {
            "api_key": "token",
        },
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    },
    "AzureGPT4":{
        "full_id": "AzureGPT4",
        "equivalence_class": "AzureOpenAI",
        "credentials": {
            "api_key": "AzureOpenAI",
        },
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    }
}
MODELS = list(MODEL_CONFIGS.keys())
# TASKS = [
#     "implicatures",
#     "metaphor_boolean",
#     "navigate",
#     "presuppositions_as_nli",
#     "sports_understanding",
#     "vitaminc_fact_verification",
#     "winowhy",
#     "word_sorting",
# ]
TASKS = [
    "GSM8K",
    # "BigBench",
    "implicatures",
    "MATH",
]
METHODS = ["sammo", "apo", "ape", "grips", "baseline"]

ACCURACY={
    "implicatures": accuracy,
    "GSM8K": accuracy_GSM8K,
    "MATH": accuracy_MATH,
}

class InstructionTuningSearchSpace:
    def __init__(self, dtrain, dincontext):
        self.dtrain = dtrain
        self.dincontext = dincontext
        self.format_search_space = [
            XMLDataFormatter(all_labels=self.dtrain.outputs.unique(), orient="item"), 
            JSONDataFormatter(all_labels=self.dtrain.outputs.unique(), orient="item"), 
            QuestionAnswerFormatter(all_labels=self.dtrain.outputs.unique(), orient="item"), 
            MultiLabelFormatter(all_labels=self.dtrain.outputs.unique(), orient="item")
            ]

    def __call__(self):
        example_formatter = PlainFormatter(all_labels=self.dtrain.outputs.unique(), orient="item")
        instructions = MetaPrompt(
            [
                Paragraph(
                    "",
                    id="instructions",
                ),
                Paragraph("Examples:\n"),
                Paragraph(FewshotExamples(self.dincontext[0])),
                Paragraph("\n"),
                Paragraph(InputData()),
                Paragraph("Output: Let's think step by step."),
            ],
            render_as="raw",
            data_formatter=example_formatter,
        )

        return Output(
            instructions.with_extractor("raise"),
            minibatch_size=1,
            on_error="empty_result",
        )

@click.command()
@click.option("--llm", default=MODELS[0], type=click.Choice(MODELS), prompt=True)
@click.option("--task", default=TASKS[0], type=click.Choice(TASKS), prompt=True)
@click.option("--method", default=METHODS[0], type=click.Choice(METHODS), prompt=True)
@click.option("--uuid", default=None, type=str)
@click.option("--confirmed", is_flag=True, default=True)
@click.option("--marker", default="test", type=str)

def main(llm, task, method, marker, uuid=None, confirmed=None, debug=False):
    if confirmed is None:
        click.confirm(f"Do you want to run {task} with {llm}?", abort=True, default=True)
    model_config = MODEL_CONFIGS[llm]
    # add current time to run_id
    from datetime import datetime
    run_id = f"{task}_{model_config['equivalence_class'].replace('/', '_')}"

    eval_config = MODEL_CONFIGS["vllm"]
    eval_runner = Vllm(
        model_id=eval_config["full_id"],
        api_config=eval_config["credentials"],
        equivalence_class=eval_config["equivalence_class"],
        rate_limit=eval_config["rate_limit"],
        cache=sammo.store.PersistentDict(MAIN_FOLDER / "cache" /f"eval_{run_id}.cache.tsv"),
        timeout=eval_config["timeout"],
        max_retries=50000,
        max_context_window=eval_config["max_context_window"],
    )

    # all_tasks = {x["task_id"]: x for x in orjson.loads(pathlib.Path(DATA).read_bytes())}
    # task = all_tasks[task_id]

    # data = dict()
    # for k, v in task.items():
    #     if k.startswith("d_"):
    #         data[k] = DataTable.from_records(v, constants=dict(instructions=task["instructions"]))

    task_data = orjson.loads(pathlib.Path(f"./data/{task}.json").read_bytes())
    data = dict()
    for k, v in task_data.items():
        if k.startswith("d_"):
            data[k] = DataTable.from_records(v)
    objective = ACCURACY[task]

    # search_space = InstructionTuningSearchSpace(data["d_train"], data["d_incontext"])
    search_space = InstructionTuningSearchSpace(data["d_train"], data["d_incontext"])
    
    if method == "baseline":
        baseline_performance = EnumerativeSearch(
            runner=eval_runner, 
            search_space=search_space, 
            objective=objective, 
            max_candidates=1)
        baseline_performance.fit_transform(data["d_train"][:10], data["d_val"][:10])
        dtest_pred = baseline_performance.transform(data["d_test"])
        baseline_performance.show_report()
        print(f"Test score: {objective(data['d_test'], dtest_pred)}")
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        baseline_performance.save_json(MAIN_FOLDER / "baseline" / f"{run_id}_{time}.model.json")

    opt_config = MODEL_CONFIGS["AzureGPT4"]
    opt_runner =AzureGPT4(
        model_id=opt_config["full_id"],
        api_config=opt_config["credentials"],
        equivalence_class=opt_config["equivalence_class"],
        rate_limit=opt_config["rate_limit"],
        cache=sammo.store.PersistentDict(MAIN_FOLDER / f"opt_{run_id}.cache.tsv"),
        timeout=opt_config["timeout"],
        max_retries=50000,
        max_context_window=opt_config["max_context_window"],
    )

    if method == "ape":
        prompt_optimizer = BeamSearch(
            runner,
            APE({"id": "instructions"}, search_space, data["d_train"], 5),
            objective,
            maximize=True,
            n_initial_candidates=12,
            depth=3,
            mutations_per_beam=2,
            beam_width=4,
            add_previous=True,
        )

    elif method == "apo":
        prompt_optimizer = BeamSearch(
            eval_runner,
            opt_runner,
            APO(
                {"id": "instructions", "_child": "content"},
                search_space,
                num_gradients=4,
                steps_per_gradient=1,
                num_rewrites=4,
                minibatch_size=5,
            ),
            objective,
            maximize=True,
            depth=20,
            mutations_per_beam=100,
            beam_width=8,
            add_previous=True,
        )
    elif method == "grips":
        mutation_operators = SyntaxTreeMutator(
            {"id": "instructions"},
            search_space,
            PersistentDict(MAIN_FOLDER / "trees" / f"{run_id}.cache.json"),
        )
        prompt_optimizer = BeamSearch(
            runner,
            mutation_operators,
            objective,
            maximize=True,
            depth=7,
            mutations_per_beam=2,
            n_initial_candidates=1,
            beam_width=4,
            add_previous=True,
        )
    elif method == "sammo":
        mutation_operators = BagOfMutators(
            search_space,
            InduceInstructions({"id": "instructions"}, data["d_incontext"]),
            APO(
                {"id": "instructions", "_child": "content"},
                None,
                num_gradients=4,
                steps_per_gradient=1,
                num_rewrites=4,
                minibatch_size=40,
            ),
            Paraphrase({"id": "instructions"}),
            ChangeDataFormat(
                {"id": "instructions"},
                choices=search_space.format_search_space),
            sample_for_init_candidates=True,
            )
        prompt_optimizer = BeamSearch(
            # runner,
            eval_runner,
            opt_runner,
            mutation_operators,
            objective,
            maximize=True,
            depth=20,
            mutations_per_beam=100,
            n_initial_candidates=1,
            beam_width=8,
            add_previous=True,
        )

    prompt_optimizer.fit(data["d_train"],data["d_val"])
    prompt_optimizer.show_report()

    if not debug:
        dtest_pred = prompt_optimizer.transform(data["d_test"])
        print(f"Test score: {objective(data['d_test'], dtest_pred)}")
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    prompt_optimizer.save_json(MAIN_FOLDER / method / f"{run_id}_{marker}_{time}.model.json")

if __name__ == "__main__":
    main()
