import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *

from src.core import file_manager


class PipelineHelper:
    def __init__(self, actor):
        self.actor = actor

        self.spark = sparknlp.start(spark32=True)

        self.path_annotated_sentences = file_manager.filename_from_data_dir(
            f'output/{self.actor}/annotated_sentences.csv'
        )

        self.annotated_sentences = self.spark.read.option("header", True).csv(
            self.path_annotated_sentences
        ).select('txt')

    def apply_embeddings(self, stages, output_columns, embedding_name, aliases=[]):
        use_clf_pipeline = Pipeline(stages=stages)

        print('training ....')
        use_pipeline_model = use_clf_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("txt"))        

        print('applying pipeline ....')
        df_embeddings = use_pipeline_model.transform(self.annotated_sentences)

        for alias in aliases:
            output_columns.append(df_embeddings[alias].alias(aliases[alias]),)
        
        df_embeddings = df_embeddings.select(output_columns)

        output_file = file_manager.filename_from_data_dir(
            f'embeddings/{embedding_name}/text_emb_{self.actor}.json'
        )

        print('saving data....')
        df_embeddings.write.json(output_file)

    def apply_sentence_embeddings(self, embedding_annotator, embedding_name):
        document_assembler = DocumentAssembler().setInputCol("txt").setOutputCol("document")

        self.apply_embeddings(
            stages=[document_assembler, embedding_annotator],
            output_columns=['txt', 'sentence_embeddings.embeddings'],
            embedding_name=embedding_name
        )

    def apply_word_embeddings(self, word_embedding, embedding_name):
        documentAssembler = DocumentAssembler().setInputCol("txt").setOutputCol("document")
        sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
        tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

        embeddings_sentenses = SentenceEmbeddings().setInputCols(['document', 'embeddings']).setOutputCol(
            'sentence_embeddings').setPoolingStrategy('AVERAGE')

        stages = [documentAssembler, sentence_detector, tokenizer, word_embedding, embeddings_sentenses]
        
        output_columns = ['txt', 'sentence_embeddings.embeddings']

        aliases = {'token.result': 'tokens', 'embeddings.embeddings': 'word_embeddings'}

        self.apply_embeddings(
            stages=stages,
            output_columns=output_columns,
            embedding_name=embedding_name,
            aliases=aliases
        )

    def generate_embedding_lasbe(self):
        embedding_annotator = BertSentenceEmbeddings.pretrained("labse", "xx") \
            .setInputCols(["document"]).setOutputCol("sentence_embeddings")

        self.apply_sentence_embeddings(embedding_annotator=embedding_annotator, embedding_name='lasbe')

    def generate_embedding_use(self):
        embedding_annotator = UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx") \
            .setInputCols(["document"]).setOutputCol("sentence_embeddings")

        self.apply_sentence_embeddings(embedding_annotator=embedding_annotator, embedding_name='use')


    def generate_embedding_glove(self):
        # documentAssembler = DocumentAssembler().setInputCol("txt").setOutputCol("document")
        # sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
        # tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

        embeddings = WordEmbeddingsModel().pretrained("glove_6B_300", "xx") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings")

        # embeddings_sentenses = SentenceEmbeddings().setInputCols(['document', 'embeddings']).setOutputCol(
        #     'sentence_embeddings').setPoolingStrategy('AVERAGE')

        # stages = [documentAssembler, sentence_detector, tokenizer, embeddings, embeddings_sentenses]

        # self.apply_embeddings(
        #     stages=stages,
        #     output_columns=['txt', 'sentence_embeddings.embeddings'],
        #     embedding_name='glove'
        # )

        self.apply_word_embeddings(word_embedding=embeddings, embedding_name='glove')

    def generate_embedding_bert_pt(self):
        bert_embeddings = BertEmbeddings.pretrained("bert_portuguese_base_cased", "pt") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings")

        self.apply_word_embeddings(word_embedding=bert_embeddings, embedding_name='bert_pt')
