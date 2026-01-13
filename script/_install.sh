set -e

echo "Installing the necessary packages ..."
pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# Prefer prebuilt wheel; fallback to editable build only if wheel fails.
if ! pip install pytorch3d==0.7.8; then
  echo "Wheel install failed, trying editable build (requires CUDA toolchain & torch present)..."
  cd third_party/pytorch3d
  pip install . --no-build-isolation
  cd ../..
fi

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien || true
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
if [ -f "$URDF_LOADER" ]; then
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
else
  echo "sapien not found, skip urdf_loader patch"
fi

echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib || true
PLANNER=$MPLIB_LOCATION/planner.py
if [ -f "$PLANNER" ]; then
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
else
  echo "mplib not found, skip planner patch"
fi

echo "Installing Curobo ..."
# if [ -d envs ]; then
#   cd envs
#   if git clone https://github.com/NVlabs/curobo.git; then
#     cd third_party/curobo
#     pip install -e . --no-build-isolation
#     cd ..
#   else
#     echo "curobo clone failed (network?), skip"
#   fi
#   cd ..
# else
#   echo "envs directory not found, skip curobo install"
# fi
cd third_party/curobo
pip install -e . --no-build-isolation
cd ..
echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
