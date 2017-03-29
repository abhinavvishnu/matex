#!/usr/bin/perl -w

use strict;

my $stf = `python3 --version`;
$stf =~ s/^\s+//;
$stf =~ s/\s+$//;
my @parts = split / /,$stf;
unless(defined($parts[$#parts])){
   die "0\n";
}
my @numbers = split /\./,$parts[$#parts];
my $val = $numbers[0] . "." . $numbers[1];
print "$val\n";

