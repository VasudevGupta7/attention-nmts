# Run this file to install all dependencies
# @author: vasudevgupta

echo installing all the dependencies

pip install -r requirements.txt

echo installing spacy models for english & german

python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

echo YAYYYYY, everything is setup; Play however way you want