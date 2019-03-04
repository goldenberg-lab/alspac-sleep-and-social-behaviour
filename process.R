# This is a preprocessing pipeline for ALSPAC data, doing the following things in order: 
#
# 1) Load the raw data from the SPSS file.
# 2) Make code replacements where possible. This only includes values that can reasonably be imputed  
#     based on the questionnaire. Makes these replacements based on the custom built mappings file 
#     called 'protocols-completed.txt'. 
# 3) For each column, replace invalid codes with 'NA'. The codes are read from a custom built list
#     called 'invalids.csv'. These codes are chosen because they are defined to be equivalent to 
#     'NA' in the ALSPAC data dictionary (questionnaire pdfs). 
# 4) Optional: For each value in a set of threshold values, get a list of features whose number of 
#     missing values do not surpass the respective threshold value. These lists are written into 
#     the 'thresholded' directory. 
# 5) Optional: Calculate missingness proportions for each variable.
# 6) Optional: Generate a count for each factor code over the entire dataset (this only applies to 
#     factor variables). 
# 7) Remove columns with too many missing values based on chosen 'missingness_threshold'. 
#     NOTE: This is only done if 'threshold_remove_missing_features' is set to TRUE. 
#     Remove columns with zero variance. Remove columns with perfect correlation with 
#     another column. Remove columns corresponding to the 'in_xxxx' variables, which denote 
#     questionnaire participation and are redundant. 
# 8) Save final data frame. 
#
library(foreign)
library(plyr) 

# flags for optional steps
threshold_missing_features <- TRUE
calculate_missingness <- TRUE 
generate_factor_counts <- FALSE 
threshold_remove_missing_features <- FALSE   # Set to TRUE to remove overly missing features
missingness_threshold <- 10000   # missing value threshold

# file paths 
spss_path <- path.expand('~/data/alspac/Goldenberg_14Nov17.sav')  # raw data 
replacements_path <- path.expand('~/alspac/workspace/varlists/protocols-completed.txt')  # replacement code mappings 
invalids_path <- path.expand('~/alspac/workspace/factor_codes/invalids.csv')  # invalid factor codes
feats_base_path <- path.expand('~/alspac/workspace/thresholded/ft')  # output feature list (per threshold)
na_counts_path <- path.expand('~/alspac/workspace/na_counts.csv')  # output NA counts for each column 
factor_codes_count_path <- path.expand('~/alspac/workspace/labels.csv')  # output for factor code counts 
frame_path <- path.expand('~/alspac/workspace/frame.Rdata')  # output data.frame 

######################### 1) LOAD RAW DATA #########################

# load data from SPSS file and convert to data.frame 
data <- read.spss(spss_path, to.data.frame=TRUE, use.value.labels=FALSE, 
                  use.missing=FALSE)
rownames(data) <- paste(data$cidB2855, data$qlet, sep="")
warnings() 

# helper function to search data frame columns for given factor label 
seekvar <- function(label) {
  df <- data.frame(Feature=character(), stringsAsFactors=FALSE)
  for(i in seq(1, ncol(data))){
    if('value.labels' %in% names(attributes(data[,i]))){
      labels <- attr(data[, i], "value.labels")
      if(label %in% names(labels)){
        n <- nrow(df) + 1
        df[n, ] <- colnames(data[i])
      }
    }
  }
  return(as.character(df$Feature))
}

######################### 2) MAKE CODE REPLACEMENTS #########################

# read table of protocols
protocol <- read.table(file=replacements_path, header=FALSE, stringsAsFactors=FALSE)
colnames(protocol) <- c("variable", "protocol")

# parse out replacements 
replacements <- protocol[grepl(":", protocol$protocol), ]

# replace factor codes
for(i in 1:nrow(replacements)){
  # get variable
  var <- replacements$variable[i]

  if(var %in% names(data)){
    # get 'before' and 'after' factor code values
    reps <- sapply(strsplit(replacements$protocol[i], ","), function(r) strsplit(r, ":"))
    origs <- unname(sapply(reps, function(r) r[1]))
    subs <- unname(sapply(reps, function(r) r[2]))

    # make replacement
    data[, var] <- mapvalues(data[, var], origs, subs)
  }
}

######################### 3) REPLACE INVALID CODES #########################

# get list of invalid factor codes from file 
invalids <- read.csv(invalids_path, header=FALSE, blank.lines.skip=FALSE)
invalids <- as.character(invalids$V1)

# encode invalid labels as NA (missing) 
for(i in seq(1, ncol(data))){ 
  # check if value.labels attribute exists
  if('value.labels' %in% names(attributes(data[,i]))){ 
    
    # get vector of labels 
    labels <- attr(data[, i], 'value.labels')
    
    # get indices of relevant labels
    x <- which(names(labels) %in% invalids)
    
    if(length(x) > 0){
      # replace with NA 
      data[, i] <- mapvalues(data[, i], c(labels[x]), 
                             rep(c(NA), length(x)), warn_missing=FALSE)
      
      # update value.labels attribute 
      attr(data[, i], 'value.labels') <- labels[!names(labels) %in% invalids]
    }
  } 
}

######################### 4) THRESHOLD MISSING VALUES #########################

if(threshold_missing_features){
  # list of threshold values 
  thresholds <- c(8000, 9000, 9500, 10000, 10500, 11000, 12000, 13000, nrow(data))

  for(thresh in thresholds){
    # remove columns with # NAs greater than 'thresh'
    chop <- data[,colSums(is.na(data)) <= thresh]

    # compute missingness fraction of entire data frame 
    miss_prop <- sum(is.na(chop)) / (nrow(chop) * ncol(chop))

    # save to file 
    f_path <- paste(feats_base_path, thresh, 
                    'n', ncol(chop), 
                    'ms', round(miss_prop, 3), sep='_')
    write(names(chop), file=f_path)
  }
}

######################### 5) MISSINGNESS PROPORTIONS #########################

if(calculate_missingness){
  # get missing value rate for each column 
  na_count <- sapply(data, function(y) sum((is.na(y))))
  
  # convert to data frame
  na_count <- data.frame(na_count)

  # add total count as extra column 
  na_count$total <- nrow(data) 

  # save 
  write.table(na_count, sep=',', file=na_counts_path, row.names=TRUE, col.names=TRUE)
}

######################### 6) GENERATE FACTOR COUNT #########################

# function to construct a data frame of factor code counts
populate_counts <- function(){
  # initialize data.frame of label counts 
  cnts_df <- data.frame(Label=character(), Value=character(), 
                        Count=numeric(), stringsAsFactors=FALSE)
  
  # populate label counts 
  for(i in seq(1, ncol(data))){
    # check if value.labels attribute exists
    if('value.labels' %in% names(attributes(data[,i]))){
      
      # get labels  
      labels <- attr(data[, i], "value.labels")
      
      # iterate labels vector
      if(length(labels) > 0) {
        for(j in seq(1, length(labels))){ 
          
          # label name, value, and count 
          label <- attr(labels[j], 'names')
          val <- labels[j][[1]]
          count <- sum(data[, i] == labels[j][[1]], na.rm=TRUE)
          
          if(label %in% cnts_df$Label & val %in% cnts_df$Value){            
            # add to existing row count 
            prev <- cnts_df[cnts_df$Label == label & cnts_df$Value == val, 'Count']
            cnts_df[cnts_df$Label == label & cnts_df$Value == val, 'Count'] <- prev + count 
            
          } else { 
            # create new row with count 
            n <- nrow(cnts_df) + 1
            cnts_df[n,] <- list(label, val, count) 
          }
        }
      }
    }
  }
  # ensure Value column is numeric
  cnts_df$Value <- as.numeric(cnts_df$Value) 
  
  # sort label count data.frame by Count column 
  cnts_df <- cnts_df[order(cnts_df$Count, decreasing=TRUE), ]
  
  return(cnts_df)
}

if(generate_factor_counts){
  # generate count data.frame
  cnts_df <- populate_counts()  
  
  # save to csv file 
  write.csv(file=factor_codes_count_path, x=cnts_df, row.names=FALSE) 
}

######################### 7) REMOVE COLUMNS #########################

# The following list of features are the DAWBA bands. They have high 
# missingness (~10,000) but can be used. Based on the DAWBA 
# questions, ALSPAC derived placements for each individual in one of 
# five risk bands for each disorder. These are used in 
# previous studies, e.g. Walton ADHD study. 
no_remove <- c('levelband_15','pextband_15','padhdbandd_15','padhdbandi_15',
                    'pbehavband_15','poddband_15','pcdband_15','semotband_15',
                    'sdepband_15','sanxband_15','sgenaband_15','spanband_15',
                    'sagoband_15','sptsdband_15','ssophband_15','sspphband_15',
                    'any01_15','pext01_15','padhd01_15','phk01_15','pbehav01_15',
                    'podd01_15','pcd01_15','semot01_15','sdep01_15','sanx01_15',
                    'sgena01_15','span01_15','sago01_15','sptsd01_15','ssoph01_15',
                    'sspph01_15')

# get current feature list
feats <- names(data)

# remove 'cid___' and 'qlet' variables as they are already in the rownames
data <- data[, !feats %in% c('cidB2855', 'qlet')]
feats <- names(data)  # update features list 

# get list of 'in_xxxx' variables from feature list 
ins <- feats[startsWith(feats, 'in_')]  

# remove 'in_xxxx' variables 
data <- data[, !feats %in% ins]
feats <- names(data)  # update features list 

# get number of unique values in each column 
uniqs <- sapply(data, function(r) length(unique(na.omit(r))))

# get features with only 1 unique value (barring NAs)
singles <- names(uniqs[uniqs == 1])
singles <- singles[!singles %in% no_remove]

# remove these features 
data <- data[, !feats %in% singles]
feats <- names(data)  # update features list 

if(threshold_remove_missing_features) {
  
  # get list of features with missing values > threshold
  remove <- feats[colSums(is.na(data)) > missingness_threshold]
  
  # ignore features in the 'no_remove' list
  remove <- remove[!remove %in% no_remove]
  
  # remove features with too many missing values from data frame 
  data <- data[, !feats %in% remove]
  feats <- names(data)  # update features list 
}

######################### 8) SAVE DATA FRAME #########################

# save data frame
save(data, file=frame_path)


