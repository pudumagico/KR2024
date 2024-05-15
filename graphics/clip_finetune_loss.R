library(tidyverse)
library(reshape2)

eval_loss <- read.csv("clip_finetune_validation_loss.csv")
training_loss <- read.csv("clip_finetune_training_loss.csv")

loss <- training_loss %>%
  left_join(eval_loss, by="Step") %>%
  rename("Training" = "Value.x", "Validation" = "Value.y") %>%
  select(c("Step", "Training", "Validation")) %>%
  melt(id.vars=c("Step")) %>%
  drop_na()

ggplot(loss, aes(x=Step, y=value, color=variable)) +
  geom_line(size=1) +
  scale_color_manual(values=c("#9B79AA", "#4C0099"))
  ylab("Loss") +
  guides(color=guide_legend(title="Dataset")) +
  theme(text = element_text(size = 13))
