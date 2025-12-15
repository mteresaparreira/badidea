# "Bad Idea, Right?"
## Exploring Anticipatory Human Reactions for Outcome Prediction in HRI

**[Maria Teresa Parreira](https://www.mariateresaparreira.com/)<sup>1</sup>, [Sukruth Gowdru Lingaraju](https://www.linkedin.com/in/glsukruth/)<sup>1</sup>, [Adolfo Ramirez-Aristizabal](https://scholar.google.com/citations?user=iF7yqHkAAAAJ&hl=en)<sup>2</sup>,[Alexandra Bremers](https://bremers.github.io/)<sup>1</sup>, [Manaswi Saha](https://manaswisaha.github.io/)<sup>2</sup>,  [Michael Kuniavsky](https://scholar.google.com/citations?user=KtPU2SMAAAAJ&hl=en)<sup>2</sup>, [Wendy Ju](https://wendyju.com/)<sup>1</sup>**

¹Cornell Tech, ²Accenture Labs

Published at the 2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN)

*For inquiries, please contact Teresa (mb2554 [at] cornell [dot] edu).*
---

## Abstract

Humans have the ability to anticipate what will happen in their environment based on perceived information. Their anticipation is often manifested as an externally observable behavioral reaction, which cues other people in the environment that something bad might happen. As robots become more prevalent in human spaces, robots can leverage these visible anticipatory responses to assess whether their own actions might be "a bad idea?"

In this study, we delved into the potential of human anticipatory reaction recognition to predict outcomes. We conducted a user study wherein 30 participants watched videos of action scenarios and were asked about their anticipated outcome of the situation shown in each video ("good" or "bad"). We collected video and audio data of the participants reactions as they were watching these videos. We then carefully analyzed the participants' behavioral anticipatory responses; this data was used to train machine learning models to predict anticipated outcomes based on human observable behavior.

Reactions are multimodal, compound and diverse, and we find significant differences in facial reactions. Model performances are around 0.5-0.6 test accuracy, and increase notably when nonreactive participants are excluded from the dataset. We discuss the implications of these findings and future work. This research offers insights into improving the safety and efficiency of human-robot interactions, contributing to the evolving field of robotics and human-robot collaboration.

---

## Study Protocol

### Overview
We conducted an online crowd-sourced study to collect webcam reactions to stimulus videos from a global sample recruited through Prolific. The study protocol included:

1. **Participants**: 30 participants (ages 20-39, diverse backgrounds)
2. **Stimulus Dataset**: 30 short videos (9.62 ± 2.77 seconds) featuring humans and robots in various action scenarios
3. **Two-Stage Design**:
   - First, participants watched a shorter version of each video (stopping before the outcome)
   - Participants then predicted: "You think this situation ends..." with options "well" or "poorly"
   - Finally, participants watched the full video showing the actual outcome

### Data Collection
- Webcam recordings at 30 fps captured participants' facial reactions
- Video order was randomized for each participant
- Participants could not see themselves during video playback
- Study duration: approximately 30 minutes

### Analysis
- Facial feature extraction using OpenFace 2.0
- 35 facial action units (AUs) analyzed
- Machine learning models tested: RNNs, LSTMs, GRUs, BiLSTMs, and Deep Neural Networks
- Statistical analysis revealed 12 action units with significantly different activation intensities between "good" and "bad" anticipated outcomes

---

## Example Reactions

Here are some examples of anticipatory reactions captured during the study:

### Responses to good (left) vs bad (right) outcome
<p align="center">
  <img src="submission/images/0_reactive.gif" width="45%" alt="Reactive Response 1">
  <img src="submission/images/1_reactive.gif" width="45%" alt="Reactive Response 2">
</p>

### Diverse Anticipatory Behaviors
<p align="center">
  <img src="submission/images/1.gif" width="30%" alt="Reaction 1">
  <img src="submission/images/2.gif" width="30%" alt="Reaction 2">
  <img src="submission/images/3.gif" width="30%" alt="Reaction 3">
</p>

<p align="center">
  <img src="submission/images/4.gif" width="30%" alt="Reaction 4">
  <img src="submission/images/5.gif" width="30%" alt="Reaction 5">
  <img src="submission/images/6.gif" width="30%" alt="Reaction 6">
</p>

### Example Stimulus
<p align="center">
  <img src="submission/images/stimulus.gif" width="60%" alt="Example Stimulus Video">
</p>

### Comparison: Reactive vs Non-Reactive
Our study found that some participants displayed very visible reactions (especially for anticipated bad outcomes), while others showed subtle to no reactions at all.

<p align="center">
  <img src="submission/images/non_reactive.gif" width="45%" alt="Non-Reactive Example">
</p>

---

## Key Findings

- **Reactions to bad outcomes are more salient**: Participants displayed more diverse and visible reactions when anticipating bad outcomes
- **Multimodal and evolving responses**: Anticipatory behaviors include facial expressions, head motion, body pose changes, and vocalizations that compound and evolve over time
- **Person-dependent variability**: Different participants showed varying degrees of reactivity
- **Significant facial features**: 12 facial action units showed significantly different activation patterns between good and bad anticipated outcomes, including:
  - Inner Brow Raiser
  - Brow Lowerer
  - Cheek Raiser
  - Nose Wrinkler
  - Lip Corner Puller
  - Jaw Drop
  - Blink
- **Model performance**: Best models achieved ~60% accuracy on the curated dataset, with notable improvements when non-reactive participants were excluded

---

## Applications

This research has implications for:
- Robot error prevention systems
- Human-robot collaboration safety
- Adaptive robot behavior based on human social cues
- Human-AI collaborative systems
- Proactive failure detection in human-robot interaction

---

## Citation

```bibtex
@INPROCEEDINGS{10731310,
  author={Parreira, Maria Teresa and Lingaraju, Sukruth Gowdru and Ramirez-Artistizabal, Adolfo and Bremers, Alexandra and Saha, Manaswi and Kuniavsky, Michael and Ju, Wendy},
  booktitle={2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN)},
  title={"Bad Idea, Right?" Exploring Anticipatory Human Reactions for Outcome Prediction in HRI},
  year={2024},
  volume={},
  number={},
  pages={2072-2078},
  keywords={Accuracy;Navigation;Human-robot interaction;Machine learning;Predictive models;Data models;Behavioral sciences;Safety;Robots;Videos;robot error;social signals;anticipation;error prevention;computer vision;human-AI collaboration},
  doi={10.1109/RO-MAN60168.2024.10731310}
}
```

---

## Resources

### Paper
Read the full paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10731310)

DOI: [10.1109/RO-MAN60168.2024.10731310](https://doi.org/10.1109/RO-MAN60168.2024.10731310)

### Supplementary Material
PDF available [here](submission/ROMAN2024_BADIdea_supp.pdf)

### Presentation Slides
View the presentation slides [here](https://docs.google.com/presentation/d/1uPxuz1ECplZpMuz2cuPoZt9bN1a6iz1TqgByaPx65gk/edit?usp=sharing)

### Stimulus Videos
Information about the stimulus dataset is available in [stimulus_dataset_information_repository.xlsx](stimulus_dataset_information_repository.xlsx)

### Code
Implementation code is available in the `code/` directory.

---

## Contact

For questions or collaborations, please reach out to the authors through their respective institutions.

**Cornell Tech Interaction Research Lab**: [https://irl.tech.cornell.edu/](https://irl.tech.cornell.edu/)
