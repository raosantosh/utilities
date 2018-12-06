MODEL_NUMBER=462000

OUTPUT=../resources/datasets/predictions/annotations_5scale_with_rawsearchterm_and_name_predictions_${MODEL_NUMBER}.csv
MODEL_NAME=../resources/models/cid-131-letters-${MODEL_NUMBER}
ANNOTATION_FILE=../resources/datasets/annotations/annotations_5scale_with_rawsearchterm_and_name.csv
cat  ${ANNOTATION_FILE} | python ../src/eval_test.py ${MODEL_NAME} > ${OUTPUT}

python ../src/perf.py ${OUTPUT}

a = query2producttripletrepresentation("action figur","action figur 2 2pack 2packbrbul 3 375 375inch 4 5 accessori accessoriesbrbrbul accessoriesbrbul  adventur against age all ani appearancebrbul awesom backdrop battl beach brace brawn brbrinclud brbrreliv brbrstation bunker children choke climact collect combat creat creatur dare delux discov do droid dual dualprojectil enter evil excit facil featur  firepow from galact galaxi generat get gigoran go good hasbro hasbrobrstar hazard headquart headtohead heavygunn hero his immers imperi inchbrbu",4,32000)
b = query2producttripletrepresentation("action figur","2 2pack 2packbrbul 3 375 375inch 4 5 accessori accessoriesbrbrbul accessoriesbrbul  adventur against age all ani appearancebrbul awesom backdrop battl beach brace brawn brbrinclud brbrreliv brbrstation bunker children choke climact collect combat creat creatur dare delux discov do droid dual dualprojectil enter evil excit facil featur  firepow from galact galaxi generat get gigoran go good hasbro hasbrobrstar hazard headquart headtohead heavygunn hero his immers imperi inchbrbu",4,32000)

print(len(a))
print(len(b))
print(len(a.difference(b)))

query = "fisher price cradl n swing"
productname = "fisher pric sweet surround monkey cradl n sw"
productdescription = "your babi will go banana for the fisher pric sweet surround monkey cradl n sw with two swing motion sid  sid  head  toe   varieti of other customiz featur includ relax vibrat you can choose  combine what your littl one like best. The adorab and ultra plush monkey seat pad is super cozi and machine washab, and the overhead mobil has three jungle friend and a mirror for your snuggle bug to admire. plus the plugin option saves on batteri so little ones can relax, swing and play in soothing comfort. where development comes into play sensory: gentle motion, soft fabric, cuddly friend overhead and soothing music and sounds help stimulate your baby developing sense. security and happiness: gentle motion, music and sounds provide comfort and securiti for your little one."
brand = "fisher price"

score(model,query,productname,productdescription,brand)


MODEL_NUMBER=75000

OUTPUT=../resources/datasets/annotations/annotations_courtney_prediction.csv
MODEL_NAME=../resources/datasets/models/cid-131-brandsampling-${MODEL_NUMBER}
ANNOTATION_FILE=../resources/datasets/annotations/annotations_courtney.csv
cat  ${ANNOTATION_FILE} | python ../src/eval.py ${MODEL_NAME} > ${OUTPUT}


MODEL_NUMBER=462000

OUTPUT=/Users/a.mantrach/Downloads/all_annotations.json_minimodel_predictions.txt
MODEL_NAME=../resources/models/cid-131-letters-${MODEL_NUMBER}
ANNOTATION_FILE=/Users/a.mantrach/Downloads/all_annotations.json_minimodel.txt
cat  ${ANNOTATION_FILE} | python ../src/eval_test.py ${MODEL_NAME} > ${OUTPUT}
python perf.py ${lsOUTPUT}

../resources/models/cid-131-letters-462000


OUTPUT=/Users/a.mantrach/Downloads/all_annotations.json_minimodel_predictions.txt
MODEL_NAME=../resources/models/cid-131-letters-${MODEL_NUMBER}
ANNOTATION_FILE=/Users/a.mantrach/Downloads/all_annotations.json_minimodel.txt
cat  ${ANNOTATION_FILE} | python ../src/eval_test.py ${MODEL_NAME} > ${OUTPUT}
python perf.py ${lsOUTPUT}