#!/usr/bin/env python
# coding: utf-8

# **GUID:** 2507608
# 
# **GitHub URL:** (https://github.com/lordofanywhere/ai-for-the-arts-a/)[https://github.com/lordofanywhere/ai-for-the-arts-a/]

# # Critically Engaging with AI Ethics
# 
# In this lab we will be critically engaging with existing datasets that have been used to address ethics in AI. In particular, we will explore the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). This challenge brought to light bias in the data that sparked the [Jigsaw Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). 
# 
# In this lab, we will dig into the dataset ourselves to explore the biases. We will further explore other datasets to expand our thinking about bias and fairness in AI in relation to aspects such as demography and equal opportunity as well as performance and group unawareness of the model. We will learn more about that in the tutorial below.
# 
# # Task 1: README!
# 
# This week, coding activity will be minimal, if any. However, as always, you will be expected to incorporate your analysis, thoughts and discussions into your notebooks as markdown cells, so I recommend you start up your Jupyter notebook in advance. As always, **remember**:
# 
# - To ensure you have all the necessary Python libraries/packages for running code you are recommended to use your environment set up on the **Glasgow Anywhere Student Desktop**.
# - Start anaconda, and launch Jupyter Notebook from within Anaconda**. If you run Jupyter Notebook without going through Anaconda, you might not have access to the packages installed on Anaconda.
# - If you run Anaconda or Jupyter Notebook on a local lab computer, there is no guarantee that these will work properly, that the packages will be available, or that you will have permission to install the extra packages yourself.
# - You can set up Anaconda on your own computer with the necessary libraries/packages. Please check how to set up a new environement in Anaconda and review the minimum list of Python libraries/packages, all discussed in Week 4 lab.
# - We strongly recommend that you save your notebooks in the folder you made in Week 1 exercise, which should have been created in the University of Glasgow One Drive - **do not confuse this with personal and other organisational One Drives**. Saving a copy of your notebooks on the University One Drive ensures that it is backed up (the first principles of digital preservation and information mnagement).
# - When you are on the Remote desktop, the `University of Glasgow One Drive` should be visible in the home directory of the Jupyter Notebook. Other machines may require additional set up and/or navigation for One Drive to be directly accessible from Jupyter Notebook.
# 

# # Task 2: Identifying Bias
# 
# This week we will make use of one of the [Kaggle](https://www.kaggle.com) tutorials and their associated notebooks to learn how to identify different types of bias. Biases can creep in at any stage of the AI task, from data collection methods, how we split/organise the test set, different algorithms, how the results are interpreted and deployed. Some of these topics have been extensively discussed and as a response, Kaggle has developed a course on AI ethics:
# 
# - Navigate to the [Kaggle tutorial on Identifying Bias in AI](https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial). 
# - In this section we will explore the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) to discover different types of biases that might emerge in the dataset. 
# 
# #### Task 2-a: Understanding the Scope of Bias
# 
# Read through the first page of the [Kaggle tutorial on Identifying Bias in AI] to understand the scope of biases discussed at Kaggle.
# 
# **- How many types of biases are described on the page?**
# 
# **- Which type of bias did you know about already before this course and which type was new to you?** 
# 
# **- Can you think of any others? Create a markdown cell below to discuss your thoughts on these questions.**
# 
# Note that the biases discussed in the tutorial are not an exhaustive list. Recall that biases can exist across the entire machine learning pipeline. 
# 
# - Scroll down to the end of the Kaggle tutorial page and click on the link to the exercise to work directly with a model and explore the data.** 
# 
# #### Task 2-b: Run through the tutorial. Take selected screenshorts of your activity while doing the tutorial.
# 
# - Discuss with your peer group, your findings about the biases in the data, including types of biases. 
# - Demonstrate your discussion with examples and screenshots of your activity on the tutorial. Present these in your own notebook.
# 
# Modify the markdown cell below to address the Tasks 2-a and 2-b.

# **Markdown for discussing bias**
# 
# The kaggle describes six types of bias:
# 
# 1. Historical bias
# 2. Representation bias
# 3. Measurement bias
# 4. Aggregation bias
# 5. Evaluation bias
# 6. Deployment bias
# 
# 
# I was aware of historical bias, although I did not know its name, or that it is also known as "garbage in, garbage out". I was also aware of representation bias and I had considered evaluation bias, although not the full extent of the problems that may arise. I was not aware of measurement bias, aggregation bias or deployment bias.
# 
# It is interesting to see how bias can creep up at every stage of a ML model, from definition to use.
# 
# Some of the biases that I would add are: automation bias, or the tendency to consider of higher quality results generated by automations over results not generated by automations; confirmation bias, where a model is built to fit a hypothesis or beliefe; and experimenter's bias, when an experimenter trains a model to produce a result that confirms their hypothesis.
# 
# 
# When I test the model, it works pretty well with insults and derogatory terms, but it does not detect toxic language if I use asterisks to replace any letter in a derogatory term.
# 
# ![testing the model](./images/screenshot1.png)
# 
# We can start seeing some bias creeping up. When I enter "I have a black friend" or "I have a muslim friend", the model recognises the messages as toxic, even though it does not recognise "white/jewish/christian" as toxic.
# 
# ![bias](./images/screenshot2.png)
# 
# 
# The course then asks questions regarding what biases would appear in hypothetical scenarios. We see historical bias in comments that refer to Islas as more likely to be classified as toxic, and measurement bias and aggregation bias on comments translated into English, as we introduce error when classifying translated comments; the model would likely perform better for comments in all languages than if comments from all languages were treated differently.
# 
# ![hypothetical scenarios](./images/screenshot3.png)
# 
# Lastly, we see how a model trained with dataset with comments from one region may not perform well on different regions, because of evaluation bias. The model would suffer from deployment bias if deployed to other regions. As some regions are over-represented, the model would suffer from representation bias, as well.
# 
# ![hypothetical scenarios](./images/screenshot4.png)

# # Task 3: Large Language Models and Bias: Word Embedding Demo
# 
# Go to the [embedding projector at tensorflow.org](http://projector.tensorflow.org/). This may take some time to load so be patient! There is a lot of information being visualised. This will take especially long if you select "Word2Vec All" as your dataset. The projector provides a visualisation of the langauge language model called **Word2Vec**.
# 
# This tool also provides the option of visualising the organisation of hand written digits from the MNIST dataset to see how data representations of the digits are clustered together or not. There is also the option of visualising the `iris` dataset from `scikit-learn` with respect to their categories. Feel free to explore these as well if you like.
# 
# For the current exercise, we will concentrate on exploring the relationships between the words in the **Word2Vec** model. First, select **Word2Vec 10K** from the drop down menu (top lefthand side). This is a reduced version of **Word2Vec All**. You can search for words by submitting them in the search box on the right hand side. 
# 
# #### Task 3.1: Initial exploration of words and relationships
# 
# - Type `apple` and click on `Isolate 101 ppints`. This reduces the noise. Note how juice, fruit, wine are closer together than macintosh, computers and atari. 
# - Try also words like `silver` and `sound`. What are your observations. Does it seem like words related to each other are sitting closer to each other?
# 
# #### Task 3.2: Exploring "Word2Vec All" for patterns
# 
# - Try to load "Word2Vec All" dataset if you can (this may take a while so be patient!) and explore the word `engineer`, `drummer`or any other occupation - what do you find? 
# - Do you think perhaps there are concerns of gender bias? If so, how? If not, why not? Discuss it with our peer group and present the results in a your notebook.
# - Why not make some screenshots to embed into your notebook along with your comment? This could make it more understandable to a broader audience. 
# - Do not forget to include attribution to the authors of the Projector demo.
# 
# Modify the markdown cell below to present your thoughts.

# **Markdown cell for discussing large language models**
# 
# **Task 3.1**
# 
# 1. Words related to each other do sit indeed close together, but there is often a dominant category. In the case of `silver`, 'gold', 'copper', 'bronze' and 'iron' are the closest words, which suggests that the dominant meaning of `silver` is that of a precious metal. However, if we take `silver` as a colour, 'purple' only appears after 'golden', and 'blue' after 'nickel', 'medal' and 'tin'.
# 
# In the case of `sound`, noise and sounds re the closes words, but I could not find any word amongst the top 101 points that considere dthe meaning of sound as 'logical' or 'reasonable', 'complete' or 'safe', or 'accepted' or 'established'.
# 
# **Task 3.2**
# 
# When exploring the "Word2Vec All" dataset, we could find some historical gender bias. For example, when exploring `drummer`, some typically male names were amongst the closer words. Those names were 'Billy', 'Dave', 'Nick', 'Johnny', 'Steve', 'Jeff', 'Lee', 'Chris', 'Phil' and 'Doug'. Interestingly, the word 'actress' was at a distance of 0.559, but it is the only female gendered word I could find.
# 
# ![gender bias](./images/screenshot5.png)
# 
# In the case of `engineer`, there no evident gendered words amongst the closest ones. This is not to say that there is no gender bias or other kind of bias in the model, but this was not evident through measuring distance.
# 
# When looking for `leader`, the historical gender bias does creep up. Words like 'chairman' (as opposed to 'chairperson' or 'chairwoman'), 'businessman' (as opposed to business person or 'businesswoman'), and wife (with no 'husband' or 'spouse' in sight) appear.
# 
# ![gender bias](./images/screenshot6.png)

# # Task 4: Thinking about AI Fairness 
# 
# So we now know that AI models (e.g. large language models) can be biased. We saw that with the embedding projector already. We discussed in the previous exercise about the machine learning pipeline, how the assessment of datasets can be crucicial to deciding the suitability of deploying AI in the real world. This is where data connects to questions of fairness.
# 
# - Navigate to the [Kaggle Tutorial on AI Fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness). 
# 
# #### Task 4-a: Topics in AI Fairness
# Read through the page to understand the scope of the fairness criteria discussed at Kaggle. Just as we dicussed with bias, the fairness criteria discussed at Kaggle is not exhaustive. 
# - How many criteria are described on the page? 
# - Which criteria did you know about already before this course and which, if any, was new to you? 
# - Can you think of any other criteria? Create a markdown cell and note down your discussion with your peer group on these questions.
# 
# #### Task 4-b: AI fairness in the context of the credit card dataset. 
# Scroll down to the end of [the page on AI fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness) to find a link to another interactive exercise to run code in a notebook using credit card application data.
# - Run the tutorial, while taking selected screenshots.
# - Discuss your findings with your peer group.
# - Note down the key points of your activity and discussion in your notebook using the example and screenshots of your activity on the tutorial.
# 
# 
# Report the results of the activity and discussion by modifying the markdown cell below.

# **Markdown cell for discussing fairness**
# 
# **1. Task 4-a**
# 
# **How many criteria are described on the page?** 
# 
# Four criteria:
# 1. Demographic parity / statistical parity
# 2. Equal opportunity
# 3. Equal accurace
# 4. Group unaware / "Fairness through unawareness"
# 
# **Which criteria did you know about already before this course and which, if any, was new to you?** I was aware of Demographic/statistical parity and equal accuracy. I was aware of group unaware, but not applied to a ML model.
# 
# I was not aware of equal opportunity as a fairness criteria.
# 
# **Can you think of any other criteria? Create a markdown cell and note down your discussion with your peer group on these questions.**
# 
# Other fairness criteria could be:
# <ul>
#     <li>Counterfactual fairness: i.e. a model is fair if for a particular individual or group its prediction in the model is the same as for a different version of the individual or group (i.e. if the individual belonged to a different demographic group).</li>
#     <li>Group fairness: makes decisions at a group level, identified by gender, ethnicity or other sensitive attributes. It can lead to incorrect outcomes which may result in discrimination.</li>
#     <li>Individual fairness: a model is fair if it produces similar decisions for similar individuals.</li>
#     <li>Preference-based fairness: given the choice between various sets of
# decision treatments or outcomes, any group of users would collectively prefer its
# treatment or outcomes, regardless of the (dis)parity as compared to the other groups (Zafar <i>et al.</i> 2017).</li>
#     <li>Equality of resources: unequal distribution of social benefits is only fair if the individuals have unequal ambitions, not if they have unequal opportunities or endowments.</li>
#     <li>Equality of capability of functioning: variations amongst individuals give them unequal powers to achieve goals even if they have the same opportunities. The playing field should be levelled to create equality of powers to take advantage of the opportunities in equal conditions (Sen 1992).</li>
#  </ul
# 

# # Task 5: AI and Explainability
# 
# In this section we will explore the reasons behind decisions that AI makes. While this is really hard to know, there are some approaches developed to know which features in your data (e.g. median_income in the housing dataset we used before) played a more important role than others in determining how your machine learning model performs. One of the many approaches for assessing feature importance is **permutation importance**.
# 
# The idea behind permutation importance is simple. Features are what you might consider the columns in a tabulated dataset, such as that might be found in a spreadsheet. 
# - The idea of permutation importance is that a feature is important if the performance of your AI program gets messed up by **shuffling** or **permuting** the order of values in that feature column for the entries in your test data. 
# - The more your AI performance gets messed up in response to the shuffling, the more likely the feature was important for the AI model.
#  
# To make this idea more concrete, read through the page at the [Tutorial on Permutation Importance](https://www.kaggle.com/code/dansbecker/permutation-importance) at Kaggle. The page describes an example to "predict a person's height when they become 20 years old, using data that is available at age 10". 
# 
# The page invites you to work with code to calculate the permutation importance of features for an example in football to predict "whether a soccer/football team will have the "Man of the Game" winner based on the team's statistics". Scroll down to the end of the page to the section "Your Turn" where you will find a link to an exercise to try it yourself to calculate the importance of features in a Taxi Fare Prediction dataset.
# 
# #### Task 1-a: Carry out the exercise, taking screenshots of the exercise as you make progress. Using screen shots and text in your notebook, answer the following question: 
# 1. How many features are in this dataset? 
# 2. Were the results of doing the exercise contrary to intuition? If yes, why? If no, why not? 
# 3. Discuss your results with your peer group.
# 4. Include your screenshots, text, and discyssions in a markdown cell.
# 
# #### Task 1-b: Reflecting on Permutation Importance.
# 
# - Do you think the permutation importance is a reasonable measure of feature importance? 
# - Can you think of any examples where this would have issues? 
# - Discuss these questions in your notebook - describe your example, if you have any, and discuss the issues. 

# **Task 1-a**
# 
# There are 8 features in the dataset.
# 
# ![dataset features](./images/permutation-importance1.png)
# 
# The most important features related to location, which was expected. Paasenger count is not relevant in most cities, and the feature was not as important in the model. However, latitude seems to matter more than longitude, which can be explained with different hypotheses, including direction of travel, price bands and tolls.
# 
# ![permutation importance](./images/permutation-importance2.png)
# 
# When measuring the distance travel, the model takes those features as more important than individual latitude or longitude, which matches intuition in most cases.
# 
# ![permutation importance](./images/permutation-importance3.png)
# 
# **Task 1-b**
# 
# Permutation importance is a useful measure of feature importance and can help debugging a model, but there are some considerations we need to make. Amongst those: in smaller datasets, there is more opportunity for luck/chance, and in those cases, random chance can cause predictions in some features to be more accurate than what the model predicts. In those cases, permutation importance could have a negative value.
# 
# It is not useful to make predictions (or generalisations) using the permutation importance results, as there are many reasons why some features are more important than others, like the example of New York taxis, where the latitudinal distance feature is more important than the longitudinal distance feature. The permutation importance measures which feature is more important, but we need to look at the different possibilities in order to hypothesise about the reasons for this.

# # Task 6: Further Activities for Broader Discussion
# 
# Apart from the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) another challenge you might explore is the [**Inclusive Images Challenge**](https://www.kaggle.com/c/inclusive-images-challenge). Read at least one of the following.
# 
# - The [announcement of the Inclusive Images Challenge made by Google AI](https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html). Explore the [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) - this is where the Inclusive Images Challenge dataset comes from.
# - Article summarising [the Inclusive Image Challenge at NeurIPS 2018 conference](https://link.springer.com/chapter/10.1007/978-3-030-29135-8_6)
# - Explore the [recent controversy](https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias) about bias in relation to [PULSE](https://paperswithcode.com/method/pulse) which, among other things, sharpens blurry images.
# - Given your exploration in the sections above, what problems might you foresee with [these tasks attempted with the Jigsaw dataset on toxicity](https://link.springer.com/chapter/10.1007/978-981-33-4367-2_81)?
# 
# There are many concepts (e.g. model cards and datasheets) omitted in discussion above about AI and Ethics. To acquire a foundational knowledge of transparency, accessibility and fairness:
# 
# - You are welcome to carry out the rest of the [Kaggle course on Intro to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) to see some ideas from the Kaggle community. 
# - You are welcome to carry out the rest of the [Kaggle tutorial on explainability]( https://www.kaggle.com/learn/machine-learning-explainability) but these are a bit more technical in nature.
# - 

# **Discussion**
# 
# Some of the problems would be:
# - The dataset that the algorithm was trained on was affected by bias. This dataset may be skewed towards a specific demographic.
# - The algorithm itself could be biased.
# - Despite technical issues, the algorithm could be adopted to make automated decisions, which may result unfair.
# - Even using correct data, underlying societal bias may creep.
# - If commercial systems do not adjust their algorithms to correct bias, contribute to perpetuating existing problems.
# 
# Applied to the Jigsaw dataset on toxicity, this could result in under-representation of some protected or minority groups (ethnicl, racial, religious, sexual orientation, gender identity, etc), which in turn may mean that comments that these groups may consider toxic are not recognised as such by the algorithm.
# 
# If the pool of researchers is not diverse, or if they do not build their algorithms to correct for diversity, they risk reproducing unconscious societal biases in their work, which can further exclude protected and minority groups, perpetuating discrimination and exclusion.

# # Summary
# 
# In this lab, you explored a number of areas that pose challenges with regard to AI and ethics: bias, fairness and explainability. This, and other topics in reposible AI development, is currently at the forefront of the AI landscape. 
# 
# The discussions coming up in the lectures on applications of AI (to be presented by guest lecturers in the weeks to come) will undoubtedly intersect with these concerns. In preparation, you might think, in advance, about **what distinctive questions about ethics might arise in AI applications in law, language, finance, archives, generative AI and beyond**.   

# In[ ]:




