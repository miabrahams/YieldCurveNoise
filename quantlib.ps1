

## Setup virtualenv
Add-PathItem "$($env:DROPBOX)/Papers/TreasuryVol/venv/Scripts\"
activate.ps1



# Run specification.
function runCRSP($args) {
    Add-PathItem "$($env:DROPBOX)/Papers/TreasuryVol/venv/Scripts/"
    activate.ps1
    cd "$($env:DROPBOX)/Papers/TreasuryVol"
    while (1) {
        python "$($env:DROPBOX)/Papers/TreasuryVol/CRSP_EstimateDaily.py" @args
    }
}


# See source code for explanation of arguments.
cls; runCRSP -b 8,10 -ew t -zr f





### Compile custom QuantLib ###


# Setup paths
$env:QL_DIR="D:\code\QuantLib\"
$env:INCLUDE="D:\code\boost_1_66_0"
Add-PathItem "D:\code\swigwin-3.0.12"

cd D:\code\QuantLib-SWIG\Python


# Remove old versions and rebuild
rm build\lib.win-amd64-3.6\QuantLib\*
rm -Recurse build\*
python setup.py clean


# Run this only for the first time or when modifying SWIG bindings
# python setup.py wrap


# Main build
python setup.py build


# Copy to venv directory2
cp build\lib.win-amd64-3.6\QuantLib\* $($env:DROPBOX)\Papers\TreasuryVol\venv\Lib\site-packages\QuantLib2





