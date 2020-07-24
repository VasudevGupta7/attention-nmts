# Seq2Seq
Implemented the seq2seq model with different approaches. 

# Goal
- [x] [Building NMTS using seq2seq architecture with Luong Attention](rnn_attention.py)
- [x] [Fully Attention based MT model](transformers.py)

# Model Architectures
## 1) Neural Machine Translation System with Attention
Implemented the seq2seq model with Luong attention for building NMTS <br>
<img src="assets/rnn_model.png" width="700" height="400"/> <br>

## 2) Neural Machine Translation System using Transformer
Implemented the Transformer for building NMTS <br>
<img src="assets/transformers_model.png" width="700" height="400"/> <br>

# Running this Project
**Clone this project using**: <br>
`git clone https://github.com/VasudevGupta7/seq2seq.git`<br>
**Run this script to install all the dependencies**: <br>
`sh script.sh`<br>
**To train the model, run the following command**:<br>
`python3 main.py`<br>
**To translate english to german execute the following command**:<br>
`python3 translate2ger.py`<br>

# References
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
