# bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase2/analysis.src \
# /export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase2/analysis.txt ./ \
# ./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase2/devtest.src \
/export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase2/devtest.txt ./ \
./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase2/train.src \
/export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase2/train.txt ./ \
./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase1/train.src \
/export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase1/train.txt ./ \
./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase1/analysis.src \
/export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase1/analysis.txt ./ \
./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase1/devtest.src \
/export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated-remove-extra/phase1/devtest.txt ./ \
./models/en-fa-52k-v1-remove-u200/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

# bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase2/train.src \
# /export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated/phase2/train.txt ./ \
# ./models/en-fa-52k-v1/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/

# bash translate.sh /export/c01/haoranxu/for_others/for_mahsa/farsi-base/phase1/train.src \
# /export/c01/haoranxu/for_others/for_mahsa/farsi-base-translated/phase1/train.txt ./ \
# ./models/en-fa-52k-v1/checkpoint_best.pt /export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/