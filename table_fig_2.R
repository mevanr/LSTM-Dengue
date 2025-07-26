
rm(list=ls())

# Load required libraries
library(tidyverse)
library(ggplot2)
library(gridExtra)

# 1. Data Input and Validation
moh_data <- data.frame(
  MOH = c("D1", "D2A", "D2B", "D3", "D4", "D5"),
  Inspected_Premises = c(3988, 3077, 5884, 7373, 5873, 1080),
  Positive_Premises = c(259, 206, 484, 554, 437, 72),
  Inspected_Wet = c(5942, 7493, 18490, 23559, 14774, 4282),
  Inspected_Dry = c(13990, 14380, 37322, 44880, 28659, 8251),
  Positive_Ae_aegypti_Wet = c(305, 308, 698, 818, 514, 108),
  Positive_Ae_albopictus_Wet = c(91, 92, 172, 188, 187, 30),
  Positive_Both_Wet = c(7, 6, 16, 23, 16, 12),
  Positive_Ae_aegypti_Dry = c(190, 161, 347, 423, 302, 58),
  Positive_Ae_albopictus_Dry = c(60, 34, 106, 106, 102, 8),
  Positive_Both_Dry = c(9, 11, 31, 25, 33, 6)
)

# Validate data integrity
stopifnot(!any(is.na(moh_data)))

# 2. Calculate Metrics with Checks
moh_data <- moh_data %>%
  mutate(
    Pct_Positive_Premises = (Positive_Premises / Inspected_Premises) * 100,
    Pct_Wet_Containers = (Inspected_Wet / (Inspected_Wet + Inspected_Dry)) * 100,
    Total_Positive_Containers = Positive_Ae_aegypti_Wet + Positive_Ae_albopictus_Wet + Positive_Both_Wet +
                                 Positive_Ae_aegypti_Dry + Positive_Ae_albopictus_Dry + Positive_Both_Dry,
    Pct_Ae_aegypti_Containers = (Positive_Ae_aegypti_Wet + Positive_Ae_aegypti_Dry) / Total_Positive_Containers * 100,
    Pct_Ae_albopictus_Containers = (Positive_Ae_albopictus_Wet + Positive_Ae_albopictus_Dry) / Total_Positive_Containers * 100,
    Pct_Both_Containers = (Positive_Both_Wet + Positive_Both_Dry) / Total_Positive_Containers * 100,
    Total_Species_Occurrences = (Positive_Ae_aegypti_Wet + Positive_Ae_aegypti_Dry) +
                                 (Positive_Ae_albopictus_Wet + Positive_Ae_albopictus_Dry) +
                                 (Positive_Both_Wet + Positive_Both_Dry),
    Pct_Premises_Ae_aegypti = (Positive_Ae_aegypti_Wet + Positive_Ae_aegypti_Dry) / Total_Species_Occurrences * 100,
    Pct_Premises_Ae_albopictus = (Positive_Ae_albopictus_Wet + Positive_Ae_albopictus_Dry) / Total_Species_Occurrences * 100,
    Pct_Premises_Both = (Positive_Both_Wet + Positive_Both_Dry) / Total_Species_Occurrences * 100
  )

# Validate sum-to-100 checks
stopifnot(
  all(rowSums(moh_data[, c("Pct_Ae_aegypti_Containers", "Pct_Ae_albopictus_Containers", "Pct_Both_Containers")]) > 99.9 &
      rowSums(moh_data[, c("Pct_Ae_aegypti_Containers", "Pct_Ae_albopictus_Containers", "Pct_Both_Containers")]) < 100.1),
  all(rowSums(moh_data[, c("Pct_Premises_Ae_aegypti", "Pct_Premises_Ae_albopictus", "Pct_Premises_Both")]) > 99.9 &
      rowSums(moh_data[, c("Pct_Premises_Ae_aegypti", "Pct_Premises_Ae_albopictus", "Pct_Premises_Both")]) < 100.1)
)

# 3. Reshape Data for Plotting
plot_premises_wet <- moh_data %>%
  select(MOH, Pct_Positive_Premises, Pct_Wet_Containers) %>%
  pivot_longer(cols = -MOH, names_to = "Metric", values_to = "Percentage") %>%
  mutate(Metric = recode(Metric,
                         "Pct_Positive_Premises" = "% Positive Premises",
                         "Pct_Wet_Containers" = "% Wet Containers"))

plot_container_species <- moh_data %>%
  select(MOH, Pct_Ae_aegypti_Containers, Pct_Ae_albopictus_Containers, Pct_Both_Containers) %>%
  pivot_longer(cols = -MOH, names_to = "Species", values_to = "Percentage") %>%
  mutate(Species = recode(Species,
                          "Pct_Ae_aegypti_Containers" = "Ae. aegypti",
                          "Pct_Ae_albopictus_Containers" = "Ae. albopictus",
                          "Pct_Both_Containers" = "Both"))

plot_premises_species <- moh_data %>%
  select(MOH, Pct_Premises_Ae_aegypti, Pct_Premises_Ae_albopictus, Pct_Premises_Both) %>%
  pivot_longer(cols = -MOH, names_to = "Species", values_to = "Percentage") %>%
  mutate(Species = recode(Species,
                          "Pct_Premises_Ae_aegypti" = "Ae. aegypti",
                          "Pct_Premises_Ae_albopictus" = "Ae. albopictus",
                          "Pct_Premises_Both" = "Both"))

# 4. Define Abstract High-Contrast Colors
metric_colors <- c(
  "% Positive Premises" = "#7FC97F",  # abstract green
  "% Wet Containers"    = "#BEAED4"   # abstract lavender
)

species_colors <- c(
  "Ae. aegypti"    = "#FDC086",  # soft orange
  "Ae. albopictus" = "#386CB0",  # deep blue
  "Both"           = "#F0027F"   # magenta pink
)

# 5. Create Abstract-Styled Plots
p1 <- ggplot(plot_premises_wet, aes(x = MOH, y = Percentage, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), width = 0.65,
           color = "black", alpha = 0.9) +
  scale_fill_manual(values = metric_colors) +
  labs(title = "A. Premises and Container Metrics", x = NULL, y = "Percentage (%)") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_y_continuous(limits = c(0, 40))

p2 <- ggplot(plot_container_species, aes(x = MOH, y = Percentage, fill = Species)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.9) +
  scale_fill_manual(values = species_colors) +
  labs(title = "B. Species Distribution in Containers", x = NULL, y = "Percentage (%)") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p3 <- ggplot(plot_premises_species, aes(x = MOH, y = Percentage, fill = Species)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.9) +
  scale_fill_manual(values = species_colors) +
  labs(title = "C. Species Distribution in Positive Premises", x = "MOH Zone", y = "Percentage (%)") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# 6. Combine and Save
combined_plot <- grid.arrange(
  p1, p2, p3,
  nrow = 3,
  heights = c(1, 1, 1.2)
)

ggsave("dengue_analysis_abstract_colors.png", combined_plot,
       width = 10, height = 12, dpi = 300, bg = "white")

message("Visualization saved as 'dengue_analysis_abstract_colors.png'")


