import argparse
import json
import nltk
import os
from pyspark import keyword_only
from pyspark.ml import UnaryTransformer, Transformer
from pyspark.ml.feature import NGram, IDF, CountVectorizer, Tokenizer
from pyspark.ml.linalg import Vector
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, explode, row_number, udf
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType
from pyspark.sql.window import Window
import re
import string


class WikiParser(UnaryTransformer):
    minParagraphs = Param(Params._dummy(), 'minParagraphs', 'minimum number of paragraphs to include')
    minCharacters = Param(Params._dummy(), 'minCharacters', 'minimum number of characters to include')

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, minParagraphs=1, minCharacters=500):
        super(WikiParser, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self._setDefault(minParagraphs=1, minCharacters=500)

    def createTransformFunc(self):
        return self._parse

    def outputDataType(self):
        return StringType()

    def validateInputType(self, inputType):
        assert inputType == ArrayType(ArrayType(StringType()))

    def _parse(self, paragraphs):
        assert len(paragraphs) > 0
        minParagraphs = self.getOrDefault(self.minParagraphs)
        minCharacters = self.getOrDefault(self.minCharacters)
        texts = []
        charCount = 0
        for i, paragraph in enumerate(paragraphs):
            if i > minParagraphs and charCount >= minCharacters:
                break
            assert isinstance(paragraph, list)
            html = ''.join(paragraph)
            text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', html)
            texts.append(text)
            charCount += len(text)
        return ' '.join(texts)


class Tokenizer(UnaryTransformer, DefaultParamsReadable, DefaultParamsWritable):
    stopwords = Param(Params._dummy(), 'stopwords', 'Stopwords to exclude', TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, stopwords=nltk.corpus.stopwords.words('english')):
        super(Tokenizer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self._setDefault(stopwords=nltk.corpus.stopwords.words('english'))

    def createTransformFunc(self):
        return self._tokenize

    def outputDataType(self):
        return ArrayType(StringType())

    def validateInputType(self, inputType):
        assert inputType == StringType()

    def _tokenize(self, text):
        if text is None:
            return []
        elif not isinstance(text, str) and not isinstance(text, unicode):
            raise ValueError('Invalid text of type {}: {}'.format(type(text), text))
        import nltk
        stemmer = nltk.stem.snowball.SnowballStemmer('english')
        stopwords = self.getOrDefault(self.stopwords)
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords]
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if not t.isnumeric()]
        tokens = [stemmer.stem(t) for t in tokens]
        return tokens


class BinaryTransformer(Transformer, HasInputCols, HasOutputCol):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(BinaryTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)


class Concat(BinaryTransformer, DefaultParamsWritable, DefaultParamsReadable):
    def _transform(self, dataset):
        concat_udf = udf(lambda x, y: x + y, ArrayType(StringType()))
        inputCol1, inputCol2 = self.getInputCols()
        return dataset.withColumn(self.getOutputCol(), concat_udf(dataset[inputCol1], dataset[inputCol2]))


class Dot(BinaryTransformer):
    def _transform(self, dataset):
        inputCol1, inputCol2 = self.getInputCols()
        dot_udf = udf(self._dot, FloatType())
        return dataset.withColumn(self.getOutputCol(), dot_udf(dataset[inputCol1], dataset[inputCol2]))

    def _dot(self, x, y):
        assert isinstance(x, Vector)
        assert isinstance(y, Vector)
        result = x.dot(y)
        if not isinstance(result, float):
            result = result.item()
        return float(result)


def checkpoint(spark, dataset, path):
    assert isinstance(spark, SparkSession)
    assert isinstance(dataset, DataFrame)
    assert isinstance(path, str)
    dataset.write.parquet(path)
    return spark.read.parquet(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--docs_path', default='data/wiki-sample/AA')
    parser.add_argument('-p', '--prepro_path', default='data/prepro')
    parser.add_argument('-q', '--queries_path', default='data/queries/sample.json')
    parser.add_argument('-o', '--output_path', default='data/output')
    parser.add_argument('-m', '--mode', choices=['prepro', 'fit', 'query'], default='prepro')
    parser.add_argument('-dl', '--docs_limit', type=int)
    parser.add_argument('-ql', '--queries_limit', type=int)
    parser.add_argument('-il', '--inverted_index_limit', type=int, default=5000)
    args = parser.parse_args()
    print('Running BigramPipeline with args: {}'.format(args))

    spark = SparkSession.builder.appName('BigramModel').getOrCreate()

    tokenIdsUdf = udf(lambda x: x.indices.tolist(), ArrayType(IntegerType()))
    tfIdfModelPath = os.path.join(args.prepro_path, 'tf_idf_model')
    docsTfIdfPath = os.path.join(args.prepro_path, 'docs_tf_idf')
    docsTokenIdsPath = os.path.join(args.prepro_path, 'docs_token_ids')
    docsBigramsPath = os.path.join(args.prepro_path, 'docs_bigrams')

    parser = WikiParser(inputCol='text', outputCol='text_parsed', minParagraphs=1, minCharacters=500)
    tokenizer = Tokenizer(inputCol='text_parsed', outputCol='unigrams')
    ngrams = NGram(inputCol='unigrams', outputCol='bigrams', n=2)
    concat = Concat(inputCols=['unigrams', 'bigrams'], outputCol='tokens')

    if args.mode == 'prepro':
        spark.sparkContext.setJobGroup('input', 'Read input data')
        docs = spark.read.json(args.docs_path)
        if args.docs_limit is not None:
            docs = docs.limit(args.docs_limit)

        spark.sparkContext.setJobGroup('parse_docs', 'Parse wiki documents')
        docsParsed = parser.transform(docs)
        docsParsed = checkpoint(spark, docsParsed, os.path.join(args.prepro_path, 'docs_parsed'))

        spark.sparkContext.setJobGroup('tokenize', 'Tokenize documents')
        docsTokenized = tokenizer.transform(docsParsed)
        docsTokenized = checkpoint(spark, docsTokenized, os.path.join(args.prepro_path, 'docs_tokenized'))

        spark.sparkContext.setJobGroup('ngrams', 'Compute bigrams')
        docsBigrams = ngrams.transform(docsTokenized)
        docsBigrams = concat.transform(docsBigrams)
        docsBigrams.write.parquet(docsBigramsPath)
    elif args.mode == 'fit':
        spark.sparkContext.setJobGroup('input', 'Read input data')
        docsBigrams = spark.read.parquet(docsBigramsPath).select('id', 'tokens')
        tf = CountVectorizer(inputCol='tokens', outputCol='tf',
                             vocabSize=10000000, minDF=2.0, minTF=3.0)
        idf = IDF(inputCol='tf', outputCol='idf')

        spark.sparkContext.setJobGroup('tf', 'Fit TF model')
        tfModel  = tf.fit(docsBigrams)
        docsTf = tfModel.transform(docsBigrams)
        docsTf = checkpoint(spark, docsTf, os.path.join(args.prepro_path, 'docs_tf'))

        spark.sparkContext.setJobGroup('idf', 'Fit IDF model')
        idfModel = idf.fit(docsTf)
        docsTfIdf = idfModel.transform(docsTf)
        docsTfIdf = docsTfIdf.select(docsTfIdf.id.alias('doc_id'), docsTfIdf.idf.alias('doc_idf'))
        docsTfIdf = checkpoint(spark, docsTfIdf, docsTfIdfPath)
        tfIdfModel = PipelineModel(stages=[tokenizer, ngrams, concat, tfModel, idfModel])
        tfIdfModel.save(tfIdfModelPath)

        spark.sparkContext.setJobGroup('docs_token_ids', 'Compute inverted index')
        docsTokenIds = docsTfIdf.select(docsTfIdf.doc_id, explode(tokenIdsUdf(docsTfIdf.doc_idf)).alias('token_id'))
        docsTokenIds.write.parquet(docsTokenIdsPath)
    elif args.mode == 'query':
        assert args.queries_path is not None

        spark.sparkContext.setJobGroup('input', 'Read input data')
        tfIdfModel = PipelineModel.load(tfIdfModelPath)
        docsTfIdf = spark.read.parquet(docsTfIdfPath)
        docsTokenIds = spark.read.parquet(docsTokenIdsPath)
        queries = spark.read.json(args.queries_path)
        if args.queries_limit is not None:
            queries = queries.limit(args.queries_limit)
        queries = queries.select(queries._id.alias('query_id'), queries.question.alias('text_parsed'))

        spark.sparkContext.setJobGroup('queries_tf_idf', 'Apply TF-IDF to queries')
        queriesTfIdf = tfIdfModel.transform(queries)
        queriesTfIdf = queriesTfIdf.select(queriesTfIdf.query_id, queriesTfIdf.tf.alias('query_tf'))
        queriesTfIdf = checkpoint(spark, queriesTfIdf, os.path.join(args.output_path, 'queries_tf_idf'))
        print('Finished query TF IDF')

        spark.sparkContext.setJobGroup('queries_token_ids', 'Compute query token IDs')
        queriesTokenIds = queriesTfIdf.select(queriesTfIdf.query_id, explode(tokenIdsUdf(queriesTfIdf.query_tf)).alias('token_id'))
        queriesTokenIds = checkpoint(spark, queriesTokenIds, os.path.join(args.output_path, 'queries_token_ids'))
        print('Finished query token IDs')

        spark.sparkContext.setJobGroup('doc_queries', 'Perform inverted index filtering')
        docQueries = docsTokenIds.join(queriesTokenIds, on='token_id').groupby('query_id', 'doc_id').count()
        window = Window.partitionBy(docQueries.query_id).orderBy(col('count').desc())
        docQueries = docQueries.withColumn('rank', row_number().over(window)) \
                        .filter(col('rank') <= args.inverted_index_limit) \
                        .select('query_id', 'doc_id')
        docQueries = checkpoint(spark, docQueries, os.path.join(args.output_path, 'doc_queries'))
        print('Finished inverted index filter')

        spark.sparkContext.setJobGroup('score', 'Perform scoring')
        docQueries = docQueries.join(docsTfIdf, on='doc_id').join(queriesTfIdf, on='query_id') \
                        .select('query_id', 'doc_id', 'query_tf', 'doc_idf')
        docQueries = Dot(inputCols=['doc_idf', 'query_tf'], outputCol='score').transform(docQueries)
        queryResults = docQueries.select('query_id', 'doc_id', 'score')
        queryResults.write.parquet(os.path.join(args.output_path, 'query_results'))
        print('Wrote output to {}'.format(args.output_path))

    spark.stop()


if __name__ == '__main__':
    main()
