# mlem
A library where I put my implementations of whatever machine learning algorithms I've been fascinated by lately

## Currently implemented
<details>
  <summary>Decision tree classifier</summary>

  `mlem.dectree.ID3Classifier` is an implementation of the [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm) with the extension of supporting any number of values for the target variable rather than just yes/no. There is a demo for it in [`tests/test_dectree.py`](tests/test_dectree.py), which outputs the decision tree generated for the classic dataset [`tests/datasets/tennis.csv`](tests/datasets/tennis.csv).
  
  The output looks like this on [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/#digraph%20tennis%20%7B%0Aoutlook%20%5Blabel%3Doutlook%2C%20shape%3Dellipse%5D%3B%0Aoutlook%20-%3E%20outlook_rain%20%20%5Blabel%3Drain%5D%3B%0Aoutlook_rain%20%5Blabel%3Dwind%2C%20shape%3Dellipse%5D%3B%0Aoutlook_rain%20-%3E%20outlook_rain_weak%20%20%5Blabel%3Dweak%5D%3B%0Aoutlook_rain_weak%20%5Blabel%3Dyes%2C%20shape%3Dplaintext%5D%3B%0Aoutlook_rain%20-%3E%20outlook_rain_strong%20%20%5Blabel%3Dstrong%5D%3B%0Aoutlook_rain_strong%20%5Blabel%3Dno%2C%20shape%3Dplaintext%5D%3B%0Aoutlook%20-%3E%20outlook_overcast%20%20%5Blabel%3Dovercast%5D%3B%0Aoutlook_overcast%20%5Blabel%3Dyes%2C%20shape%3Dplaintext%5D%3B%0Aoutlook%20-%3E%20outlook_sunny%20%20%5Blabel%3Dsunny%5D%3B%0Aoutlook_sunny%20%5Blabel%3Dhumidity%2C%20shape%3Dellipse%5D%3B%0Aoutlook_sunny%20-%3E%20outlook_sunny_normal%20%20%5Blabel%3Dnormal%5D%3B%0Aoutlook_sunny_normal%20%5Blabel%3Dyes%2C%20shape%3Dplaintext%5D%3B%0Aoutlook_sunny%20-%3E%20outlook_sunny_high%20%20%5Blabel%3Dhigh%5D%3B%0Aoutlook_sunny_high%20%5Blabel%3Dno%2C%20shape%3Dplaintext%5D%3B%0A%7D):
  
  ![GraphViz rendering of said tree](https://github.com/atzuur/mlem/assets/99679220/65b8cc56-d751-441a-acd0-86b6d8d311bb)
</details>

## The name
`mlem` stands for **M**achine **L**earning **E**xperiments **M**edley, and it's also [a noise that cats sometimes make](https://www.youtube.com/watch?v=kvxCU_lQwKM) :3
