println( "Εισάγετε μια λέξη παρακαλώ:")
input = readline()
if startswith(input,("από"))
   println("Ευχαριστώ για τη συμμετοχή, αλλά δεν είναι αυτή η λέξη που ζητούσα")
   println("Ήταν η λέξη που εισήγαγες ρήμα;")
   input2= readline()
    if startswith(input2,("ναι"))
        println("το ρήμα σου ξεκινά με το παραγωγικό μόρφημα \"από\"")
    else 
        println("Ευχαριστω, αλλά ενδιαφερομαι μόνο για ρήματα")
    end 
end

str="""This is a test string with punctuation marks! And also. (parentheses)"""
str2= split(str)
join(str2, " ")
x= "!" & "," & "."

str2 = split(replace(str, '.' => " ", '!' => " ", ',' => " ")," ")




function fizzbuzz(x::Int)
    is_divisible_by_three = x % 3 == 0
    is_divisible_by_five = x % 5 == 0
    if is_divisible_by_three & is_divisible_by_five
        return "fizzbuzz"
    elseif is_divisible_by_three
        return "fizz"
    elseif is_divisible_by_five
        return "buzz"
    else
        return "else"
    end
end

const LABELS = ["fizz", "buzz", "fizzbuzz", "else"]; 
features(x) = float.([x % 3, x % 5, x % 15])
features(x::AbstractArray) = hcat(features.(x)...)
getdata() = features(1:100), onehotbatch(fizzbuzz.(1:100), LABELS) 
(X, y) = getdata()


