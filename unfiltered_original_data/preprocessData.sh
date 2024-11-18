awk 'NR == 1 || /^"1/ || /^1/' CHSD_*.csv > preprocessed_CHSD.csv
awk -F, 'NR > 1 && $8 == "1" && $9 != "MA" {print "1," $3}' SWSR_SexComment.csv > preprocessed_SWSR.csv

awk -F',' '
    BEGIN { OFS="," }

    # Process the header row to identify column positions
    NR==1 {
        for (i=1; i<=NF; i++) {
            if ($i == "label") label_col = i
            else if ($i == "TEXT") text_col = i
        }
        # Print the new header
        print "label","text"
        next
    }

    # Process subsequent rows
    {
        # Check if the label column exists and equals 1
        if ($label_col == 1) {
            # Initialize the text with the TEXT column
            text = $text_col
            # Concatenate any additional fields that are part of TEXT
            for (i = text_col + 1; i <= NF; i++) {
                text = text "," $i
            }
            # Print the label and the concatenated text
            print $label_col, text
        }
    }
' COLD_*.csv > preprocessed_COLD.csv


awk 'NR == 1 || /^"1/ || /^1/' preprocessed_*.csv > combined_preprocessed.csv




