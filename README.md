<div align="center">
  <h1 style="font-size: 40px;">AlignXplore</h1>
  <p>Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals</p>
</div>

# Links

- 📜 [Paper](https://arxiv.org/abs/2505.18071v1)
- 🤗 [Data](https://huggingface.co/datasets/JinaLeejnl/AlignXplore)

# Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

# Training

## Cold-start training

```train
cd cold-start training
./sft.sh
```

## Reinforcement learning

The code is developed based on [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero).

### Train with $R_{jud}$

```train
cd reinforcement learning
./run_ppo_jud.sh
```

### Train with $R_{gen}$

Modify the file `/reinforcement learning/orz/ppo/actors.py`:
- Change line [1027](https://github.com/JinaLeejnl/AlignXplore/blob/5c5c47fa804a1a55274e5dcdeeabc40f685a18f3/reinforcement%20learning/orz/ppo/actors.py#L1027) to `RewardRayActor = ray.remote(num_gpus=1)(genRewardRayActorBase)`.

```train
cd reinforcement learning
./run_ppo_gen.sh
```
