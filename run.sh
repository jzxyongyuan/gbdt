echo "gen data..."
mkdir -p ./data
python   ./build_data.py > ./data/train.txt
python   ./build_data.py > ./data/test.txt

echo ""
echo "make..."
make > make.log 2>make.log.wf
cat make.log.wf

echo ""
echo "run..."
# ulimit -c unlimited
rm core.* log/* -f
# ./train_debug -d conf -f train.conf -i data/train.txt -m data/gbdt.mdl
# ./test_debug -d conf -f test.conf -i data/train.txt -m data/gbdt.mdl

./train -d conf -f train.conf -i data/train.txt -m data/gbdt.mdl
./test -d conf -f test.conf -i data/test.txt -m data/gbdt.mdl
