FROM jupyter/all-spark-notebook
RUN git clone https://github.com/joshuacook/document-clustering.git
RUN wget http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.1/stanford-corenlp-3.9.1-models.jar
RUN conda install --yes -c johnsnowlabs spark-nlp
