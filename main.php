<?php

use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\WordCountVectorizer;

require_once __DIR__ . '/vendor/autoload.php';

$rawSampels = [
    'Invoice #123' => false,
    'picture of my cat' => true,
    'does pineapple belong on pizza?' => false,
    'Order summary' => false,
    'Cats are better than dogs' => true,
    'How is your cat doing?' => true,
];

$dataset = Labeled::build(
    array_map(fn($str) => [$str], array_keys($rawSampels)),
    array_map(fn($important) => $important ? 'about cats' : 'not about cats', array_values($rawSampels)),
);

$multibyteTextNormalizer = new MultibyteTextNormalizer();
$wordCountVectorizer = new WordCountVectorizer(5);
$tfIdfTransformer = new TfIdfTransformer();
$dataset
    ->apply($multibyteTextNormalizer)
    ->apply($wordCountVectorizer)
    ->apply($tfIdfTransformer);

print_r($dataset);
print_r($wordCountVectorizer);
// print_r($tfIdfTransformer);

$validationDataSet = Unlabeled::build([
    ['this email is about a cat'],
    ['boring marketing newsletter'],
    ['have you seen my turtle?'],
    ['what the cat doin'],
    ['email email email'],
])->apply($multibyteTextNormalizer)->apply($wordCountVectorizer)->apply($tfIdfTransformer);

print_r($validationDataSet);

$learner = new KNearestNeighbors();
$learner->train($dataset);

$label = $learner->predict($validationDataSet);

echo "Predictions: " . implode(', ', $label) . "\n";
