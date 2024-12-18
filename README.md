HeckmanCD's Code

Data: https://github.com/bigdata-ustc

Paper Link: https://doi.org/10.1145/3627673.3679648

Our Works: http://staff.ustc.edu.cn/~qiliuql/

Abstract：
Cognitive diagnosis, a fundamental task in education assessments,
aims to quantify the proficiency level of students based on the historical
test logs. However, the interactions between students and
exercises are incomplete and even sparse, which means that only a
few exercise scores of a student are observed. A key finding is that
the pattern of this missingness is non-random in a way that could
induce bias in the estimated proficiency value. To this end, we formulate
cognitive diagnosis with a sample selection problem where
observations are sampled through non-random probabilities that
correlate with both the response correctness and features of the student
and exercise.We proposed a simple but effective method called
HeckmanCD, adapting the Heckman two-stage approach to mitigate
this endogeneity issue. We first employ an interaction model
to predict the occurrence probability of a specific student-exercise
pair. After that, a selection variable, derived from this interaction
model, is incorporated as a controlled independent variable in the
cognitive diagnosis framework. Our analysis reveals that the vanilla
estimations of the item response theory model are inherently biased
in the existence of confounders, and our method can correct this
bias by capturing the covariance. The proposed HeckmanCD can
be applied to most existing cognitive diagnosis models, particularly
deep models, and the empirical evaluation demonstrates the
effectiveness of our method while no other auxiliary information
is required such as textual descriptions of exercises.

Cite this paper: 
[1]. Han D, Liu Q, Lei S, et al. HeckmanCD: Exploiting Selection Bias in Cognitive Diagnosis[C]//Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024: 768-777.
