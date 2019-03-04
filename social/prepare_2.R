library(foreign)
library(plyr) 

# file paths
frame_path <- path.expand('~/alspac/workspace/frame.Rdata')
social_path <- path.expand('~/alspac/social/workspace/variables.txt')
out_path <- path.expand('~/alspac/social/workspace/social_data.csv')
dawba_out_path <- path.expand('~/alspac/social/workspace/kr_dawba_91_months.csv')

# load 'data' (data.frame)
load(frame_path)

# load sleep variables
social_vars <- read.csv(social_path, header=TRUE)

# filter social data 
social_data <- data[, names(data) %in% social_vars$ID]

# replace -1's with NA
rm_negone <- c('kj605', 'kj607', 'kj612', 'kj617', 'kj620', 'kj624',
               'kj628', 'kj630', 'kj642')
for(col in rm_negone){
  social_data[, col] <- mapvalues(social_data[, col], c('-1'), c(NA))
}

# replace 9's with NA
rm_nine <- c('kl762')
for(col in rm_nine){
  social_data[, col] <- mapvalues(social_data[, col], c('9'), c(NA))
}


# show summaries
for(col in names(social_data)){
  print(col)
  print(summary(as.factor(social_data[,col])))
  print(attr(social_data[,col], 'value.labels'))
}


# save final data as csv
write.csv(file=out_path, x=social_data, row.names=TRUE)


# filter DAWBA diagnoses in KR questionnaire (91 months)
dawba_vars <- c('kr800', 'kr801', 'kr802', 'kr803', 'kr803a', 'kr810', 
                'kr811', 'kr812', 'kr813', 'kr813a', 'kr815', 'kr820',
                'kr821', 'kr822', 'kr823', 'kr824', 'kr825', 'kr826',
                'kr827', 'kr827a', 'kr830', 'kr831', 'kr832', 'kr832a')

dawba_data <- data[, names(data) %in% dawba_vars]


# show dawba summaries
for(col in names(dawba_data)){
  print(col)
  print(summary(as.factor(dawba_data[,col])))
  print(attr(dawba_data[,col], 'value.labels'))
}

# save dawba data as csv
write.csv(file=dawba_out_path, x=dawba_data, row.names=TRUE)






