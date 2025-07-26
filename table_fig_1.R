# Load required packages
library(tidyverse)
library(RColorBrewer)

# Input data
df <- tribble(
  ~Premise, ~Label, ~Total_Pos_Containers, ~Water_Storing_Tanks, ~Plastic_Containers, ~Concrete_Wet_Floor, ~Gutters, ~Tyres, ~Ornamentals_Flowers,
  ~Temp_Removed_Items, ~Discarded_Degradable, ~Discarded_NonDegradable, ~Waste_Disposed_Area, ~Covering_Material, ~Non_Used_Cisterns,
  ~Religious_Containers, ~Under_Basin_Rack, ~Construction_Materials, ~Ant_Pet_Cups, ~AC_Fridge_Filters, ~Natural,

  "Houses", "26,144 (5.4%)", 1204, 1.2, 28.0, 7.1, 1.7, 3.7, 5.8, 4.7, 1.7, 4.0, 2.0, 2.1, 2.4, 0.6, 28.1, 2.7, 1.2, 1.0, 2.2,
  "Commercial Sites", "260 (19.6%)", 120, 0.8, 16.7, 5.0, 0.0, 20.0, 14.2, 2.5, 0.8, 6.7, 2.5, 1.7, 0.8, 0.0, 15.8, 5.8, 1.7, 2.5, 2.5,
  "Government Institutes", "306 (71.9%)", 595, 3.4, 12.9, 5.5, 1.5, 19.7, 3.4, 22.0, 10.9, 4.2, 1.5, 1.8, 3.0, 0.5, 3.4, 3.0, 0.5, 1.5, 1.2,
  "Private Institutes", "46 (45.7%)", 121, 0.8, 9.9, 3.3, 0.8, 35.5, 14.9, 4.1, 3.3, 4.1, 2.5, 1.7, 2.5, 0.0, 9.1, 1.7, 1.7, 1.7, 2.5,
  "Construction Sites", "62 (82.3%)", 755, 8.6, 5.4, 48.9, 0.0, 1.2, 2.9, 5.7, 1.2, 3.3, 0.8, 1.2, 1.6, 0.8, 17.8, 0.1, 0.0, 0.3, 0.4,
  "Open/Bare Land", "178 (33.7%)", 137, 0.0, 6.6, 8.1, 0.0, 7.4, 10.3, 5.9, 3.7, 19.9, 8.8, 18.4, 0.7, 0.7, 3.7, 0.7, 0.0, 0.0, 5.1,
  "Schools", "167 (77.8%)", 357, 6.4, 9.2, 6.4, 1.9, 6.4, 20.3, 3.3, 5.0, 1.9, 6.4, 12.3, 1.1, 3.9, 3.9, 1.4, 1.1, 1.1, 7.8,
  "Religious Places", "112 (63.4%)", 296, 5.1, 18.6, 6.1, 0.7, 4.4, 15.2, 5.4, 3.0, 5.7, 1.4, 4.4, 1.4, 5.4, 6.8, 7.8, 1.7, 1.7, 5.4
)

# Update x-axis label to include total positive containers
df <- df %>%
  mutate(Premise_Label = paste0(Premise, "\n(n=", Total_Pos_Containers, ")"))

# Pivot to long format
df_long <- df %>%
  pivot_longer(
    cols = -c(Premise, Label, Total_Pos_Containers, Premise_Label),
    names_to = "ContainerType",
    values_to = "Percentage"
  ) %>%
  mutate(ContainerType = str_replace_all(ContainerType, "_", " ") %>% str_to_title())

# Define manually contrasting colors (20-class qualitative)
custom_colors <- c(
  "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
  "#a6761d", "#666666", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
  "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"
)

# Apply same ggplot but with custom fill scale
ggplot(df_long, aes(x = Premise_Label, y = Percentage, fill = ContainerType)) +
  geom_col(position = "fill", width = 0.7, color = "white", size = 0.2) +
  geom_text(
    data = df, aes(x = Premise_Label, y = 1.02, label = Label),
    inherit.aes = FALSE, size = 3.5, fontface = "italic", hjust = 0.5
  ) +
  scale_y_continuous(labels = scales::percent_format(), expand = expansion(mult = c(0, 0.08))) +
  scale_fill_manual(values = custom_colors, name = "Container Type") +
  labs(
    title = "Relative Contribution of Aedes-Positive Container Types by Premise",
    subtitle = "Bar label = Number of inspected premises (Positivity %); X-axis = Premises with total positive containers",
    x = "Premise Type",
    y = "Proportion of Positive Containers"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "right",
    legend.key.size = unit(0.5, "cm"),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5)
  )
