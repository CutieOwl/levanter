umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

VENV=~/venv310
# if the venv doesn't exist, make it
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv at $VENV"
    python3.10 -m venv $VENV
fi

source $VENV/bin/activate

#pip install -U pip
#pip install -U wheel

# jax
#pip install -U "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

#echo $VENV > levanter-midi/infra/venv_path.txt

#cd levanter-midi

#pip install -e .

#pip install -U pyrallis
#pip install -U torch torchvision torchaudio

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"
