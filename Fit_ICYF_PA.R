# R commands for testing inverse function fit of ICYF partisan asymmetry, even/odd

# A is PA values for even n
A <- structure(list(n = c(2, 4, 6, 8, 
	10, 12, 14, 16, 18, 
	20, 22, 24, 26, 28, 
	30, 32, 34, 36, 38, 
	40, 42, 44, 46, 48, 
	50, 52), 
PA = c(0.25, 0.1354, 0.0937, 0.0719, 0.0585, 
	0.0493, 0.0426, 0.0376, 0.0336, 0.0304, 
	0.0278, 0.0255, 0.0237, 0.022, 0.0206, 
	0.0194, 0.0183, 0.0173, 0.0164, 0.0156, 
	0.0149, 0.0142, 0.0136, 0.0131, 0.0126, 
	0.0121)), .Names = c("n", "PA"), 
row.names = c(2L, 4L, 6L, 8L, 10L, 12L, 14L, 16L, 18L, 20L, 
22L, 24L, 26L, 28L, 30L, 32L, 34L, 36L, 38L, 40L, 42L, 44L, 
46L, 48L, 50L, 52L), class = "data.frame")

# B is PA values for odd n
B <- structure(list(n = c(3, 5, 7, 9, 
	11, 13, 15, 17, 19, 
	21, 23, 25, 27, 29, 
	31, 33, 35, 37, 39, 
	41, 43, 45, 47, 49, 
	51, 53), 
PA = c(0.0556, 0.0417, 0.0327, 0.0269, 
	0.0228, 0.0199, 0.0176, 0.0158, 
	0.0143, 0.0131, 0.0121, 0.0112, 
	0.0104, 0.0098, 0.0092, 0.0087, 
	0.0082, 0.0078, 0.0074, 0.0071, 
	0.0068, 0.0065, 0.0063, 0.006, 
	0.0058, 0.0056)), .Names = c("n", "PA"), 
row.names = c(3L, 5L, 7L, 9L, 11L, 13L, 15L, 17L, 19L, 
21L, 23L, 25L, 27L, 29L, 31L, 33L, 35L, 37L, 39L, 41L, 
43L, 45L, 47L, 49L, 51L, 53L), class = "data.frame")

# Generate array of values for n from 1 to 53
nValues <- seq(1, 53, 1)

# Try inverse function fit for even n PA values
attach(A)
names(A)

inverseA.model <- lm(1/PA ~ n)
print(summary(inverseA.model))

# See if it makes more sense to fit 1/n as a linear function of PA
inverseAv2.model <- lm(1/n ~ PA)
print(summary(inverseAv2.model))

PA.inverse2 <- 1/predict(inverseA.model,list(n=nValues))
plot(n, PA, pch=16)
lines(nValues, PA.inverse2, lwd=2, col="red", xlab="No. Districts", ylab="Partisan Asymmetry")

invisible(readline(prompt="Press [enter] to continue"))

# Try inverse function fit for odd n PA values
attach(B)
names(B)

inverseB.model <- lm(1/PA ~ n)
print(summary(inverseB.model))

# See if it makes more sense to fit 1/n as a linear function of PA
inverseBv2.model <- lm(1/n ~ PA)
print(summary(inverseBv2.model))

PA.inverse2 <- 1/predict(inverseB.model,list(n=nValues))
plot(n, PA, pch=16)
lines(nValues, PA.inverse2, lwd=2, col="red", xlab="No. Districts", ylab="Partisan Asymmetry")