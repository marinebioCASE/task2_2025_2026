# BIODCASE 2025-2026 Task 2 : Supervised detection of strongly-labelled Antarctic blue and fin whale calls 

# Scientific context

<a name="scientific_context"></a>

Antarctic blue (*Balaenoptera musculus intermedia*) and fin (*Balaenoptera physalus quoyi*) whales were nearly wiped out
during industrial whaling. For the past twenty-five years, long-term passive acoustic monitoring has provided one of the
few cost-effective means of studying them on their remote feeding grounds at high latitudes around the Antarctic
continent.

Long term acoustic monitoring efforts have been conducted by several nations in the Antarctic, and in recent years this
work has been coordinated internationally via the Acoustic Trends Working Group of the Southern Ocean Research
Partnership of the International Whaling Commission (IWC-SORP). Some of the overarching goals of the Acoustic Trends
Project include “using acoustics to examine trends in Antarctic blue and fin whale population growth, abundance,
distribution, seasonal movements, and behaviour” [1].

Within the IWC-SORP Acoustic Trends Project relevant ecological metrics include presence of acoustic calls over time
scales ranging from minutes to months. Furthermore, recent work has highlighted additional value that can be derived
from estimates of the number of calls per time-period [2].

In 2020, the Acoustic Trends Project released publicly one of the largest annotated datasets for marine bioacoustics,
the so-called *AcousticTrends_BlueFinLibrary* (ATBFL) [3]. Release of this annotated library was intended to help
standardise analysis and compare the performance of different detectors across the range of locations, years, and
instruments used to monitor these species. It has already been exploited in several benchmarking research
papers [3][4][5].

# Task definition

The task is a classical supervised multi-class and multi-label sound event detection task using strong labels, which
depict the start and end time of the events. The target of the systems is to provide not only the event class but also
the event time localization given that multiple events can be present in an audio recording (see Fig 1). In the context
of the IWC-SORP Acoustic Trends Project described above, this task is applied to the detection of 7 different call types
from Antarctic blue and fin whales, grouped together into 3 categories for evaluation. This task aims to challenge and
assess the generalization ability of models to adapt and perform in varying acoustic environments, reflecting the
real-world variability encountered in marine mammal monitoring.

<figure>
    <div class="row row-centered">
        <div class="col-xs-10 col-md-8 col-centered">
            <img src="fig1.png" class="img img-responsive">
            <figcaption>Figure 1: Overview of the sound event detection task. Blue whale (Bm) D and Z calls are present, as well as Fin whale (Bp) 20 Hz Pulse with (Bp20p) or without (Bp20) overtone and its 40Hz downsweep (BpD).</figcaption>
        </div>
    </div>
</figure>
<br>

# Call description

The 7 calls types to detect are as follows:

- Antarctic blue whale Z-Call (label: bmz) : smooth transition of single unit presence from 27 to 16 Hz, composed of
  three-parts A, B and C
- Antarctic blue whale A-Call (label: bma) : Z-call containing only the A part
- Antarctic blue whale B-Call (label: bmb) : Z-call containing only the B part
- Antarctic blue whale D-Call (label: bmd) : comprises downsweeping frequency component between 20 and 120 Hz, can also
  comprise more frequency modulations (e.g., upsweep at start)
- Fin whale 20 Hz Pulse without overtone (label: bp20) : comprises downsweeping frequency component from 30 to 15 Hz
- Fin whale 20 Hz Pulse with overtone (label: bp20plus) : 20 Hz pulse with a secondary energy at variyng frequencies (80
  to 120 Hz variation)
- Fin whale 40Hz downsweep (label : bpd) : downsweeping vocalisation ending around 40 Hz, usually below 90 Hz and above
  30 Hz

See more detailed information and examples of call
spectrograms [here](https://docs.google.com/presentation/d/1LjQ5tt_UZD4WBVVVIpkV5LljITiNK3XL/edit?usp=sharing&ouid=104566963275162196540&rtpof=true&sd=true).

# Model development

## Train and validation datasets

The overall development dataset is composed of the entire IWC-SORP ATBFL dataset, already introduced in
the [scientific context](#scientific_context). As described in Table 1, it is organized in 11 site-year deployments
located all around Antarctica, with time periods of recording ranging from 2005 to 2017. It contains a total of 6591
audio files, totaling 1880 hours of recording, sampled at 250 Hz.

The training dataset is composed of all site-year deployments with the exception of Kerguelen 2014, Kerguelen 2015 and
Casey 2017, which have been left out from training to form the validation dataset. This makes a total of 6004 audio
files for the training dataset over 8 site-year deployments, and 587 audio files for the validation dataset over 3
site-year deployments.

| Deployment           | 	Number of audio recordings	 | Total audio duration  (h)	 | Positive duration (min:s)	 | Total sound event	  |
|----------------------|------------------------------|----------------------------|----------------------------|---------------------| 
| ballenyisland2015	   | 205	                         | 204                        | 	02:46.32                  | 		2222            	 | 
| casey2014	           | 	194	                        | 	194	                      | 	14:12.00	                 | 	6866	              |
| elephantisland2013	  | 	2247	                       | 	187	                      | 	16:06.58	                 | 	21966	             |
| elephantisland2014	  | 	2595	                       | 	216	                      | 	28:08.25	                 | 	20964	             | 
| greenwich2015	       | 	190	                        | 	32	                       | 	02:03.11	                 | 	1128	              |
| kerguelen2005	       | 	200	                        | 	200	                      | 	03:31.37	                 | 	2960	              |
| maudrise2014	        | 	200	                        | 	83	                       | 	05:42.45	                 | 	2360	              |
| rosssea2014	         | 	176	                        | 	176	                      | 	08:51.00	                 | 	104	               |
| **TOTAL TRAIN**	     | 	6004	                       | 	1292	                     | 	72:40.20	                 | 	58570	             | 
| casey2017	           | 	187	                        | 	187	                      | 	06:06.42	                 | 	3263	              |
| kerguelen2014	       | 	200	                        | 	200	                      | 	11:23.02	                 | 	8822	              |
| kerguelen2015	       | 	200	                        | 	200	                      | 	07:23.30	                 | 	5542	              |
| **TOTAL	VALIDATION** | 	587	                        | 	587	                      | 	24:53.14	                 | 	17627	             |

<center>Table 1: Description of the development dataset, including details of the training and validation datasets.</center>


A more complete version of this table is
available [here](https://docs.google.com/spreadsheets/d/19DeQK-1pIP7FO6fvDA4DfAwJrfKD3FCzf7WUF9G3COo/edit?gid=0#gid=0),
with more statistics within the different classes and more information on the recording deployments.

## Annotation

### Description

The annotation data of the development dataset correspond to the one published for IWC-SORP ATBFL dataset, where each
site-year deployment comes with its own annotation file. The 11 csv annotation files were named after each corresponding
site-year deployment. Each annotated sound event is defined by the tuple (label, low_frequency, high_frequency,
start_datetime, end_datetime), with label taking a unique name in {bma, bmb, bmz, bmd, bpd, bp20, bp20plus}.

### Protocol and feedbacks

The development dataset was annotated by a group of bioacoustician experts, with one expert per site-year deployment,
following the [SORP annotation guideline](https://docs.google.com/document/d/1AGDAlIFtAInKbo3LGng2HeKwv_shXiKX/edit).
Despite such precautions, and as in most bioacoustics annotated datasets, the annotation sets might still contain some
defects that any model developer using this corpus should be aware of. Especially, despite the use of a common protocol,
ensuring sufficient consistency between the different annotators, and thus between the different site-year annotation
sets, is made particularly difficult for the following reasons, which would have required more standardized procedures:

- different annotation styles and practices: different experts have different thresholds for when they mark a call. They
  also have different styles for marking starts and ends and high and low frequencies. Also, some analysts are precise
  and some are fast, impacting the overall accuracy of bounds - few are both;
- fragmentation, multipath and splitting vs lumping - Long tonal calls can be fragmented due to propagation and
  multipath. Some analysts are splitters and annotate every fragment independently. Others are lumpers and will mark a
  single long call for all the fragments and multipaths ;
- multipath: there is often confusion among expert analysts about whether a potential multipath is an echo or a separate
  animal calling. The broader context and sequence of calls will likely be helpful here.

All these causes are of course highly worsened within time periods with low SNR. For example, the site-year deployment
Elephant island 2015 has already been recognized as a more difficult dataset to process due to this reason.

Having said that, it is widely recognized that having high quality annotations on such a large scale dataset is a very
complex and cumbersome process, both in terms of human resources and scientific expertise. As recognized in related
audio processing fields [6], these potential defects in the annotations of the development set should be seen as an
intrinsic component of this data challenge reflecting real-life annotation practices and that should be fully addressed
by models.

## Download

Raw audio recordings of the development dataset, along with annotation data, can be downloaded from
this [Zenodo entry](https://zenodo.org/records/15092732?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhkMzVmYzFiLWFjM2ItNDA0ZS04MmU2LTE2MzQ4OTZhNmI0MCIsImRhdGEiOnt9LCJyYW5kb20iOiJhMjE4NDQyNTBlYjFiYmMzOWM0MzczZmYwMDEzNzI5NCJ9.9oDVeRP8kTzfkvLXQIXlNPtC-Lf2o6dOK7RFZAjRj0bX6obwMFH_yMPIZ23z-KQxQcHTTVeDVh7GSVmeVsxdgA).
Note that minor changes were made to the original ATBFL dataset such as adopting more consistent naming conventions,
pooling together all annotation files per call type into a single file per dataset etc (see the complete list of minor
changes on Zenodo).

## Supplementary resources

* All datasets, annotations and pre-trained models from
  this [list](https://docs.google.com/spreadsheets/d/1uoPP6tNvj2WtCdjGyOFmHrmBoQx-BK8lwTTFCiCNXrg/edit?gid=0#gid=0) are
  allowed ;
* Use of other external data (e.g. audio files, annotations) and pre-trained models are allowed only after approval from
  the task coordinators (contact: dorian.cazau@ensta.fr). These external data and models should be public, open
  datasets.

# Model evaluation

## Dataset

The evaluation dataset is composed of two new site-year deployments, not yet published as parts of the ATBFL dataset.
They contain the same cetacean species as in the development dataset but from different sites in Antarctica and/or
different time periods. Those datasets will be used as independent evaluation sets to get more detailed insights into
the generalization performance of models, and an overall evaluation scoring will also be computed to have the global
ranking of models.

## Annotation

Within the IWC-SORP ATBFL project, the same annotation setup as the development dataset has been used for the evaluation
dataset. In addition to that, to ensure the highest quality of evaluation annotations, a two-phase multi-annotator
campaign has been specially designed for this challenge, including a complete re-annotation of all evaluation data, plus
a double-checking procedure of the most conflictual cases. The complete protocol and associated results will be released
at the end of the data challenge.

## Metrics

The evaluation metrics used is a 1D IoU (standing for Intersection over Inion), which basically looks at all the time
when the predicted event overlaps with the ground truth event, divided by the total time spanning from the minimum start
time to the maximum end time. To emphasize the importance of estimating an accurate number of calls (for example, for a
downstream task of population density estimation), this metrics was customized to penalize model outputs with several
detections overlapping with one single ground truth. For example, if 3 predicted sound events overlap with one single
ground truth event, only one of the predicted sound events will be marked as a true positive (TP) and assigned as
correct, and the rest will be marked as false positives (FP). TP are then computed counting all the prediction events
which have been marked as correct. FP are all the prediction events which were not assigned to a ground truth. FN are
all the ground truth events which have not been assigned any prediction. Recall, Precision and f1-score are then
computed per-class and per-deployment.

See more details in the [GITHUB repository](https://github.com/marinebioCASE/task2_2025/tree/main/evaluation)

## Download

This dataset will be released on the 1st June 2025.

## Rules and submission

Participants are free to employ any preprocessing technique and network architecture. The only requirement is that the
final output of your model MUST be a CSV file formatted following the annotation format of evaluation set described
above.

Official challenge submission consists of:

- System output file (*.csv)
- Metadata file (*.yaml)
- Technical report explaining in sufficient detail the method (*.pdf)

System output should be presented as a single text-file, in CSV format, with a header row as shown in the example output
below:

| dataset       | filename                    | annotation | start_datetime                   | end_datetime                     |
|---------------|-----------------------------|------------|----------------------------------|----------------------------------|
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2014-02-18T21:32:03.876700+00:00 | 2014-02-18T21:32:13.281600+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2014-02-18T21:37:42.187800+00:00 | 2014-02-18T21:37:51.400800+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmb        | 2014-02-18T21:39:06.640300+00:00 | 2014-02-18T21:39:15.277500+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmz        | 2014-02-18T21:48:19.270900+00:00 | 2014-02-18T21:48:28.292000+00:00 |

# Baseline

An off-the-shelf object detector model YOLOv11 was run to get baseline performance on the validation set (see table
below). In addition to providing reference performance for the task, this baseline has also been prepared to serve as a
getting started code example, where you will find some general routines to load audio and annotation files in different
Python frameworks, compute spectrograms, train baseline models and use the evaluation code on the validation set.

| Model   | Recall | Precision | f1   | 
|---------|--------|-----------|------|
| YOLOv11 | 0.32   | 0.67      | 0.43 |

See more details in the [GITHUB repository](https://github.com/marinebioCASE/task2_2025/tree/main/baselines)

# Support

If you have questions please use the BioDCASE Google
Groups [community forum](https://groups.google.com/g/biodcase-community), or directly contact task coordinators (
dorian.cazau@ensta.fr).

# References

[1] Miller et al. (2021). An open access dataset for developing automated detectors of Antarctic baleen whale sounds and
performance evaluation of two commonly used detectors, Sci. Rep., 11, 806. doi:10.1038/s41598-020-78995-8 <br>
[2] Castro et al (2024). Beyond counting calls: estimating detection probability for Antarctic blue whales reveals
biological trends in seasonal calling. Front. Mar. Sci. doi:10.3389/fmars.2024.1406678 <br>
[3] Miller et al. (2020). An annotated library of underwater acoustic recordings for testing and training automated
algorithms for detecting Antarctic blue and fin whale sounds. doi: 10.26179/5e6056035c01b <br>
[4] Schall et al. (2024). Deep learning in marine bioacoustics: a benchmark for baleen whale detection. Remote Sens Ecol
Conserv, 10: 642-654. https://doi.org/10.1002/rse2.392 <br>
[5] Dubus et al. (2024). Improving automatic detection with supervised contrastive learning: application with
low-frequency vocalizations. Workshop DCLDE (2024) <br>
[6] Fonseca et al. (2022) FSD50K: An Open Dataset of Human-Labeled Sound Events. IEEE/ACM TALP, vol. 30 (1) <br>
