# -*- coding: utf-8 -*-
from .thomas.query_generation import QueryGeneration
from .thomas.dans_from_proto import DansFromProto
from .thomas.dans_from_fasttext import DansFromFastText
from .thomas.dans_from_triplets import DansFromTriplets

all_experiments = {
    "thomas.supervised_query_generation": QueryGeneration(),
    "thomas.dans_from_proto": DansFromProto(),
    "thomas.dans_from_fasttext": DansFromFastText(),
    "thomas.dans_from_triplets": DansFromTriplets()
}
