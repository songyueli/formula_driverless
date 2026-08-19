#include "geodetic.hpp"

#include <cmath>

namespace
{
// WGS84 ellipsoid constants -- same model gz-sim's <spherical_coordinates>
// uses (surface_model=EARTH_WGS84).
constexpr double kSemiMajorAxis = 6378137.0;           // meters
constexpr double kFlattening = 1.0 / 298.257223563;
constexpr double kEccentricitySquared = kFlattening * (2.0 - kFlattening);

double DegToRad(double _deg)
{
    return _deg * M_PI / 180.0;
}

// Prime-vertical radius of curvature at the given latitude.
double PrimeVerticalRadius(double _latRad)
{
    const double sinLat = std::sin(_latRad);
    return kSemiMajorAxis / std::sqrt(1.0 - kEccentricitySquared * sinLat * sinLat);
}

// Geodetic (lat/lon/alt) -> Earth-Centered-Earth-Fixed Cartesian.
void ToEcef(double _latRad, double _lonRad, double _altM,
            double *_x, double *_y, double *_z)
{
    const double n = PrimeVerticalRadius(_latRad);
    const double cosLat = std::cos(_latRad);
    *_x = (n + _altM) * cosLat * std::cos(_lonRad);
    *_y = (n + _altM) * cosLat * std::sin(_lonRad);
    *_z = (n * (1.0 - kEccentricitySquared) + _altM) * std::sin(_latRad);
}
} // namespace

namespace fsd
{

GeodeticConverter::GeodeticConverter(double _refLatDeg, double _refLonDeg, double _refAltM)
    : m_refLatRad(DegToRad(_refLatDeg)), m_refLonRad(DegToRad(_refLonDeg))
{
    ToEcef(m_refLatRad, m_refLonRad, _refAltM, &m_refEcefX, &m_refEcefY, &m_refEcefZ);
}

EnuPosition GeodeticConverter::ToEnu(double _latDeg, double _lonDeg, double _altM) const
{
    double x, y, z;
    ToEcef(DegToRad(_latDeg), DegToRad(_lonDeg), _altM, &x, &y, &z);

    const double dx = x - m_refEcefX;
    const double dy = y - m_refEcefY;
    const double dz = z - m_refEcefZ;

    const double sinLat = std::sin(m_refLatRad);
    const double cosLat = std::cos(m_refLatRad);
    const double sinLon = std::sin(m_refLonRad);
    const double cosLon = std::cos(m_refLonRad);

    // Standard ECEF -> ENU rotation about the reference point.
    return EnuPosition{
        -sinLon * dx + cosLon * dy,
        -sinLat * cosLon * dx - sinLat * sinLon * dy + cosLat * dz,
        cosLat * cosLon * dx + cosLat * sinLon * dy + sinLat * dz};
}

} // namespace fsd
