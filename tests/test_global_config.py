from nanochat.global_config import GlobalConfig


def test_load_from_yaml_supports_hierarchical_paths(tmp_path):
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        """
device: cpu
dataset:
  root: data
  pretrain: pretrain/sample
  sft: sft/identity.jsonl
  simple_spelling: sft/words.txt
  eval: eval
  allenai_arc: ai2_arc
  openai_gsm8k: gsm8k
  openai_humaneval: humaneval
  cais_mmlu: mmlu
  huggingface_tb_smol_smoltalk: smol-smoltalk
checkpoint:
  root: ckpts
  base: base
  chatsft: sft
  chatrl: rl
output:
  root: out
  base_eval: eval
  tokenizer: tokenizer
  report: report
enforce_eager: true
""".strip(),
        encoding="utf-8",
    )

    config = GlobalConfig.load_from_yaml(str(config_path))

    assert config.pretrain_dataset == "data/pretrain/sample"
    assert config.sft_dataset == "data/sft/identity.jsonl"
    assert config.eval_dataset == "data/eval"
    assert config.output_dir == "out"
    assert config.base_checkpoints_dir == "ckpts/base"
    assert config.chatsft_checkpoints_dir == "ckpts/sft"
    assert config.chatrl_checkpoints_dir == "ckpts/rl"
    assert config.base_eval_dir == "out/eval"
    assert config.tokenizer_dir == "out/tokenizer"
    assert config.report_dir == "out/report"
