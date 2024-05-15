library(tidyverse)

run <- read.csv("testdev_base_clipbase_owllarge.csv")
run_short <- run %>% filter(runtime_sec < 4)

ggplot(run_short, aes(x=runtime_sec)) + 
  geom_histogram(binwidth = 0.1, fill="#9B79AA", color="#E1D5E7") +
  xlab("Runtime (seconds)") +
  ylab("# Questions") + 
  theme(text = element_text(size = 13)) 
