# Task 1. Natural Language Processing. Named entity recognition

In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts.

There are semantic problems:

What is considered a mountain? Are the Himalayas a mountain or not? Three Bald Heads Hill?

For example, Mount Saser Kangri has official peaks: Saser Kangri I, Saser Kangri II, Saser Kangri III, etc.

Sometimes the spurs of a mountain or its peaks are considered distinctive mountains: Lungnak La by Hrten Nyima

Let's take what Wikipedia considers mountains.

There are linguistical problems:

If the mountain is shortened in the text: Changabang -> Chang

Or if one mountain has different names and we don't know these alternative names? 'Saltoro Kangri', 'Peak 36' or 'Saser Kangri', 'Sasir Kangri'.

For now we will use the names of mountains and their synonyms found on Wikipedia.

# Solution

The BERT model is already trained to find tokens. Among them there are tokens that indicate a place.

If we run texts about mountains through the BERT model, we get “location” labels in relation to mountains.

If we compare the received location labels with the list of mountains, we will quickly find the mountains we need.

There is no need to waste time and resources for additional training of the BERT model, which, moreover, may be of questionable quality due to the linguistic and semantic problems listed above.

# Conclusions.
We compared two options: select location from tokens, and then look for mountain names in them.

And you can simply search for mountain names using all selected tokens.

In this case, the search for all tokens was a little slower. But the efficiency of mountain detection has also increased by almost 30%:

# Options for improving the accuracy of mountain recognition.

We got not only the names of the mountains in the text.

We received a labeled dataset on which the BERT model can be further trained, which can increase the accuracy of mountain recognition.

We can take plus or minus 10-15 words from the found mountain names. And train the model only on this data. We will shorten the dataset and remove unrecognized mountain names from it. This will remove noise and help the model focus on identifying patterns.

We can use ruled-based tokenizers or conditional random fields and increase the accuracy of searching for mountain names using words that occur naturally nearby.