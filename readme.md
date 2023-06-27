# 強化学習を利用したオセロAI

## Caution
このリポジトリは強化学習のアルゴリズムの勉強用のリポジトリです。利用する際は以下の点に注意してください

- 速度を考えると、実用的にはRay RLlibやKeras-RLなどの既存のライブラリを利用を推奨します。
- 対戦ゲームは相手の手番を考える必要があり実装が諸々めんどくさいので、学習用途ではおすすめしません。
- 本リポジトリでは、学習目的で適用範囲が広いモデルフリーの学習手法を用いていますが、オセロは完全情報ゲームであるため、精度を求めるならばMCTSのようなモデルベース手法が有力です。  
  cf. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf

## Website
https://othello-rl-k62eblwddq-de.a.run.app/
(ランダムモデルに対して703勝, 269敗, 28引き分け)

## 利用技術
### Q学習

Q値を用いて、状態と行動のペアに対して、そのペアの価値を表現する。

```math
Q(s,a) = Q(s,a) + \alpha (r + \gamma \max_{a'} Q(s',a') - Q(s,a))
```

### DQN

Q学習をニューラルネットワークを用いて行う。

```math
y = r + \gamma \max_{a'} Q(s',a')
```

### 経験再生

replay memoryを用いて、学習データを蓄積し、ランダムに学習データを取り出すことで、学習の効率化・データ間の相関の低減を図る。

### Target Network

学習データの相関を低減するために、2つのネットワークを用意し、片方のネットワークで学習を行い、もう片方のネットワークで評価を行うことで、学習を安定化させる。

```math
y = r + \gamma \max_{a'}Q_\text{target}(s',a')
```

### Double DQN

Q値の最大化を行う際に、Q値を最大化する行動を選択するのではなく、Q値を最大化する行動を選択するネットワークを別に用意し、そのネットワークで行動を選択することで、Q値の過大評価を防ぐ。

```math
y = r + \gamma Q_\text{target}(s',\mathop{\arg \max}_{a'} Q(s',a'))
```

### Dueling Network

Q値を状態価値と行動価値に分解し、それぞれを計算することで、学習の安定化を図る。

```math
Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a')
```
