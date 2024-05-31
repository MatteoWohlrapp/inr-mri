#!/bin/bash

# Directory to search in, default to current directory
directory=${1:-"."}

# Count files containing "FLAIR"
flair_count=$(find "$directory" -type f -name "*FLAIR*" | wc -l)
echo "Number of files with 'FLAIR' in their name: $flair_count"

# Count files containing "T1" but not "T1POST" or "T1PRE"
t1_count=$(find "$directory" -type f -name "*T1*" \
    | grep -E -v "T1POST|T1PRE" \
    | wc -l)
echo "Number of files with 'T1' (but not 'T1POST' or 'T1PRE') in their name: $t1_count"

# Count files containing "T1POST"
t1post_count=$(find "$directory" -type f -name "*T1POST*" | wc -l)
echo "Number of files with 'T1POST' in their name: $t1post_count"

# Count files containing "T1PRE"
t1pre_count=$(find "$directory" -type f -name "*T1PRE*" | wc -l)
echo "Number of files with 'T1PRE' in their name: $t1pre_count"

# Count files containing "T2"
t1pre_count=$(find "$directory" -type f -name "*T2*" | wc -l)
echo "Number of files with 'T2' in their name: $t1pre_count"
