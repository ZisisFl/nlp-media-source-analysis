## Exploratory analysis of press bias in Greek language

### Motivation
Press bias and fake news detection have been emerging topics that the NLP community has been trying to tackle in recent years. Especially in times that major political events are taking place, like the US elections, there is a lot of conversation going on, regarding the quality and authenticity of news. A good example that showcases the impact of this problem, is [Twitter’s latest mechanism](https://www.dw.com/en/how-does-twitters-tweet-labeling-work/a-53622684) that tries to label posts on the basis of their trustworthiness. Regarding Greece’s case, according to Reporters without Borders study ([2020 World Press Freedom Index](https://rsf.org/en/ranking)), Greece ranks 65th in the world and 24th among 27 European countries. To the best of our knowledge there hasn’t been a related NLP work for news in greek language. Our study focuses on detecting political bias in political news articles coming from various greek press sources, that we collected through their official web pages.

### Set up this project
You will need: 
* Python 3.6 or higher

Create a python virtual environment and activate it then install requirements:

```sh
python3 -m venv press_bias_env
source press_bias_env/bin/activate
pip install -r requirements.txt
```

### Original articles data ###
The original articles have been removed from this repo for copyright issues. You can find the tool which we created and used to collect them [here](https://github.com/foukonana/news_media_scrappers).
