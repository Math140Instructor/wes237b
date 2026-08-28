#! /bin/bash
#
# Builds the zip for Gradescope.  run_autograder only copies these five files out
# of a submission and takes everything else from its own checkout, so there is
# no reason to upload the rest -- and Eigen plus the MNIST data are far too big
# to upload anyway.

set -e

cd "$(dirname "$0")"

FILES="
lab/convolution-2d/main.c
lab/convolution-2d/kernel.cl
lab/blas_mm/main.cc
homework/cnn/src/layer/custom/new-forward.cc
homework/cnn/src/layer/custom/new-forward-kernel.cl
"

name=$(basename "$PWD")
out=$PWD/$name.zip
force=${1:-}

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

missing=0
for file in $FILES; do
	if [ ! -f "$file" ]; then
		echo "missing $file -- that part scores zero" >&2
		missing=1
		continue
	fi

	mkdir -p "$stage/$name/$(dirname "$file")"
	cp "$file" "$stage/$name/$file"

	# Leaving the marker above finished code is fine, so this is only a hint
	# that a part was never started.
	if grep -q '//@@' "$file"; then
		echo "note: $file still has //@@ markers" >&2
	fi
done

if [ "$missing" -ne 0 ] && [ "$force" != "-f" ]; then
	echo "nothing zipped; pass -f to submit anyway" >&2
	exit 1
fi

# The paths matter, not the top folder: the grader finds each part by folder
# name anywhere under the upload, then copies the listed files from the same
# relative paths inside it (so cnn keeps its src/layer/custom/... structure).
rm -f "$out"
(cd "$stage" && zip -rqX "$out" "$name")

echo "wrote $out"
unzip -Z1 "$out" | grep -v '/$' | sed 's/^/  /'
