# This is a preprocessing pipeline for Sleep data.

library(foreign)
library(plyr) 

# file paths 
frame_path <- path.expand('~/alspac/workspace/frame.Rdata')  # complete data.frame 
sleep_path <- path.expand('~/alspac/workspace/categories/sleep.csv')  # sleep variables
time_path <- path.expand('~/alspac/workspace/categories/time.csv')  # date/time variables
catalogue_path <- path.expand('~/alspac/workspace/features/csv/catalogue_ft_15445_n_4214_ms_0.582.csv')  # variable catalogue
sleep_cat_path <- path.expand('~/alspac/workspace/features/sleep_catalogue.csv')  # sleep catalogue
time_cat_path <- path.expand('~/alspac/workspace/features/time_catalogue.csv')  # time catalogue
sleep_out_path <- path.expand('~/alspac/sleep/workspace/sleep.csv')  # output sleep data
time_out_path <- path.expand('~/alspac/sleep/workspace/times.csv')  # output time mappings
frame_out_path <- path.expand('~/alspac/sleep/workspace/frame.Rdata')  # data.frame output 

# load 'data' (data.frame)
load(frame_path)

# load sleep variables
sleep_vars <- read.csv(sleep_path, header=TRUE)

# get sleep data 
sleep <- data[, names(data) %in% sleep_vars$Variable]

# get variable catalogue 
catalogue <- read.csv(catalogue_path, header=TRUE) 

# sleep variables in catalogue
sleep_cat <- catalogue[catalogue$name %in% names(sleep),]

# write sleep catalogue to csv
write.csv(file=sleep_cat_path, x=sleep_cat, row.names=FALSE)

# time variables in catalogue
time <- read.table(time_path, header=FALSE)
time <- as.character(time$V1)
time_cat <- catalogue[catalogue$name %in% time,]

# time data
time_data <- data[, names(data) %in% time]

# write time catalogue to csv
write.csv(file=time_cat_path, x=time_cat, row.names=FALSE)

# time mappings (varying by individual)
kk <- 'kk998a'  # age (in months)
kn <- 'kn9991a' # ''
kq <- 'kq998a'  # ''
kr <- 'kr991a'  # '' 
ku <- 'ku991a'  # '' 
kv <- 'kv9991a'  # ''
tb <- 'tb9991a'  # '' 
cct <- 'cct9991a'  # '' 
ccu <- 'CCU9991'  # age (in years) 

# default time mappings
time_map = list(
  kk = 54,
  kn = 69,
  kq = 81,
  kr = 91,
  ku = 108, 
  kv = 120, 
  tb = 156,
  cct = 216, 
  ccu = 240
)

# make factor code replacements and removals 

remove_factor_code <- function(df_col, code){
  labels <- attr(df_col, 'value.labels')  
  labels <- labels[names(labels) != code]
  attr(df_col, 'value.labels') <- labels
  return(df_col)
}

replace_factor_code <- function(df_col, code){
  labels <- attr(df_col, 'value.labels')
  df_col <- mapvalues(df_col, c(labels[code]), c(NA), 
                 warn_missing=FALSE)
  attr(df_col, 'value.labels') <- labels[names(labels) != code]
  return(df_col)
}

sleep$kr204 <- remove_factor_code(sleep$kr204, 'No attachments')
sleep$kr206 <- remove_factor_code(sleep$kr206, 'No attachments')
sleep$kr293 <- remove_factor_code(sleep$kr293, 'No stressor')
sleep$kr413 <- remove_factor_code(sleep$kr413, 'No moods')

sleep$kr204a <- replace_factor_code(sleep$kr204a, 'No attachments')
sleep$kr206a <- replace_factor_code(sleep$kr206a, 'No attachments')
sleep$kr293a <- replace_factor_code(sleep$kr293a, 'No stressor')


# remove sleep data rows with all NA's
sleep <- sleep[rowSums(is.na(sleep)) < ncol(sleep), ]


# generate time map as data.frame 
tm <- list() 
for(cname in names(sleep)){
  
  first <- substr(cname, 1, 1)
  first_two <- substr(cname, 1, 2)
  
  if(first == 'k' || first == 't'){
    tm[cname] <- time_map[[first_two]]
  } 
  else if(tolower(first_two) == 'cc') {
    first_three <- substr(cname, 1, 3)
    first_three <- tolower(first_three)
    tm[cname] <- time_map[[first_three]]
  } 
}
tm <- as.data.frame(unlist(tm)) 
names(tm) <- c('time') 
tm[['variable']] <- row.names(tm)
rownames(tm) <- NULL 


# combine kn2011a (hour) and kn2011b (minutes) and convert to hours
sleep$kn2011a <- sleep$kn2011a + (sleep$kn2011b / 60)

# drop sleep variable
sleep <- sleep[, names(sleep) != 'kn2011b']


# variable types:

# ordinal --> kk470, kk480, kn2010, kn2122, kq280, kr204, kr206, 
#     kr293, ku762, kv4023, kv4024, kv4025, kv5538, kv6555, tb4023,
#     tb4024, tb4025, tb5538, tb6555
# binary -->kn2000, kr204a, kr206a, kr293a, kr413, kv7034, tb7034,
#     cct5140, CCU2050c, CCU3430, CCU3431, CCU3432
# time (continuous, bounded) --> kn2011a
# categorical --> kn2034, cct5141, cct5142
# integer (positive, unbounded) --> kq317


# print sleep factor codes
for(name in names(sleep)){
  print(name)
  print(attributes(sleep[[name]]))
  print(summary(as.factor(sleep[[name]])))
}


# save final sleep data as csv, save time mappings as csv 
write.csv(file=time_out_path, x=tm, row.names=FALSE)
write.csv(file=sleep_out_path, x=sleep, row.names=FALSE)

# also save as data.frames
save(tm, sleep, file=frame_out_path)



