#include "RotatedRect.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
Line::Line(const FLOAT_TYPE m, const FLOAT_TYPE b)
// -----------------------------------------------------------------------------------
{
    this->m = m;
    this->b = b;
}

// -----------------------------------------------------------------------------------
Line::Line(const cv::Point2f &p1, const cv::Point2f &p2)
// -----------------------------------------------------------------------------------
{
    this->m = FLOAT_TYPE(p1.y-p2.y) / FLOAT_TYPE(p1.x-p2.x);
    this->b = FLOAT_TYPE(p1.y) - this->m*p1.x;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::Y(const FLOAT_TYPE x) const
// -----------------------------------------------------------------------------------
{
    return this->m * x + this->b;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::M() const
// -----------------------------------------------------------------------------------
{
    return this->m; 
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::B() const
// -----------------------------------------------------------------------------------
{
    return this->b;
}

// -----------------------------------------------------------------------------------
Line Line::perpLinePassPoint(const Line &perp_to, const cv::Point &p1)
// -----------------------------------------------------------------------------------
{
    const FLOAT_TYPE m = static_cast<FLOAT_TYPE>(-1.0 / (perp_to.M() + std::numeric_limits<FLOAT_TYPE>::epsilon()));
    const FLOAT_TYPE b = static_cast<FLOAT_TYPE>(p1.y - m * p1.x);
    return Line(m, b);
}

// -----------------------------------------------------------------------------------
#if defined(_MSC_VER) && (_MSC_VER <= 1800) // fix MSVC partial implementation of constexpr
    const FLOAT_TYPE RotatedRect::inner_rect_threshold = static_cast<FLOAT_TYPE>(0.01);
#endif
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect()
// -----------------------------------------------------------------------------------
{
    this->hack_flash = false;
    this->pA.x = this->pB.x = this->pC.x = this->pD.x = 0;
    this->pA.y = this->pB.y = this->pC.y = this->pD.y = 0;
    this->rect_up = true;
    this->half_perimeter = 0.f;
}

// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect(const cv::Rect& cv_rect)
// -----------------------------------------------------------------------------------
{
    this->hack_flash = false;
    this->pD = cv::Point2f(cv_rect.tl()) + cv::Point2f(0, static_cast<float>(cv_rect.height));
    this->pC = this->pD + cv::Point2f(static_cast<float>(cv_rect.width), 0);
    const FLOAT_TYPE height = static_cast<FLOAT_TYPE>(cv_rect.height);

    *this = RotatedRect(this->pD, this->pC, height, true);
}


// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect(const cv::Point2f &po_1, const cv::Point2f &po_2,
                         const FLOAT_TYPE height, const bool rect_up)
// -----------------------------------------------------------------------------------
{

    this->hack_flash = false;
    this->rect_up = rect_up;

    this->pD = po_1;
    this->pC = po_2;

    //std::cerr<<pD.x<<"=="<<pC.x<<" AND "<<pD.y<<"=="<<pC.y<<std::endl;

    if(std::abs(pD.x-pC.x) < 0.0001)
    {
        pD.x += 0.0001f;
        hack_flash = true;
    }
    if(std::abs(pD.y - pC.y) < 0.0001)
    {
        pD.y += 0.0001f;
        hack_flash = true;
    }

    this->lCD = Line(this->pC, this->pD);
    this->lBC = Line::perpLinePassPoint(this->lCD, this->pC);
    this->lDA = Line::perpLinePassPoint(this->lCD, this->pD);

    if (this->rect_up)
        this->lAB = Line(this->lCD.M(), lCD.B()-height*std::sqrt(lCD.M()*lCD.M()+1));
    else
        this->lAB = Line(this->lCD.M(), lCD.B()+height*std::sqrt(lCD.M()*lCD.M()+1));

    const FLOAT_TYPE xA = (lDA.B()-lAB.B()) / (lAB.M() - lDA.M());
    const FLOAT_TYPE xB = (lBC.B()-lAB.B()) / (lAB.M() - lBC.M());
    this->pA = cv::Point2f(xA,lAB.Y(xA));
    this->pB = cv::Point2f(xB,lAB.Y(xB));

    this->half_perimeter = static_cast<FLOAT_TYPE>( cv::norm(this->pA-this->pB) + cv::norm(this->pB-this->pC) );
}

// -----------------------------------------------------------------------------------
cv::Vec3d RotatedRect::SumOver(const LineIntegralImage &verticalLineIntegralImage_) const
// -----------------------------------------------------------------------------------
{
    const cv::Mat& verticalLineIntegralImage = verticalLineIntegralImage_.data;

    std::vector<cv::Point2f> ups, dos;

    if(this->pA.x <= this->pB.x && this->pA.x <= this->pC.x && this->pA.x <= this->pD.x)
    {
        // A is leftmost
        for(int x = static_cast<int>(pA.x); x < static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
        }

        for (int x = static_cast<int>(pD.x); x < static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for (int x = static_cast<int>(pB.x); x < static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }
    }

    if(this->pB.x<this->pA.x && this->pB.x<=this->pC.x && this->pB.x<=this->pD.x  )
    {
        // B is leftmost
        for(int x = static_cast<int>(pB.x); x<static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pC.x); x<static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pA.x); x<static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
        }
    }

    if ( this->pC.x<this->pB.x && this->pC.x<this->pA.x && this->pC.x<=this->pD.x  )
    {
        // C is leftmost
        for(int x = static_cast<int>(pC.x); x<static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pB.x); x<static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pD.x); x<static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }
    }

    if ( this->pD.x<this->pB.x && this->pD.x<this->pC.x && this->pD.x<this->pA.x  )
    {
        // D is leftmost
        for(int x = static_cast<int>(pD.x); x < static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pA.x); x < static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pC.x); x < static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
        }
    }

    // Do the Sum
    if (verticalLineIntegralImage.type() == CV_32SC3)
    {
        cv::Vec3d output(0, 0, 0);

        // TODO: can be parallelized
        for(auto c=0; c<dos.size(); ++c)
        {
            if(std::abs(dos[c].x-ups[c].x) > half_perimeter || std::abs(dos[c].y-ups[c].y) > half_perimeter)
                continue;

            if(dos[c].x<0 || dos[c].y<0 || dos[c].x >= verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if(ups[c].x<0 || ups[c].y<0 || ups[c].x >= verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<cv::Vec3d>(dos[c]) - verticalLineIntegralImage.at<cv::Vec3d>(ups[c]) );
        }

        return output;
    }
    else
    if (verticalLineIntegralImage.type()==CV_32SC1)
    {
        double output = 0.0;

        #pragma omp parallel for reduction(+:output)
        for(auto c = 0; c<dos.size(); ++c)
        {
            if(std::abs(dos[c].x-ups[c].x) > half_perimeter || std::abs(dos[c].y-ups[c].y) > half_perimeter)
                continue;

            if(dos[c].x < 0 || dos[c].y < 0 || dos[c].x>=verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if(ups[c].x < 0 || ups[c].y < 0 || ups[c].x>=verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<int>(dos[c]) - verticalLineIntegralImage.at<int>(ups[c]) );
        }
        return cv::Vec3d(output, output, output);
    }
    else
    if (verticalLineIntegralImage.type()==CV_64FC1)
    {
        double output = 0.0;
        
        #pragma omp parallel for reduction(+:output)
        for(auto c = 0; c < dos.size(); ++c)
        {
            if (std::abs(dos[c].x - ups[c].x) > half_perimeter || std::abs(dos[c].y - ups[c].y) > half_perimeter)
                continue;

            if (dos[c].x < 0 || dos[c].y < 0 || dos[c].x >= verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if (ups[c].x < 0 || ups[c].y < 0 || ups[c].x >= verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<double>(dos[c]) - verticalLineIntegralImage.at<double>(ups[c]) );
        }

        return cv::Vec3d(output, output, output);
    }

    return cv::Vec3d(0.0, 0.0, 0.0);
}

// -----------------------------------------------------------------------------------
cv::RotatedRect RotatedRect::GetCVRotatedRect() const
// -----------------------------------------------------------------------------------
{
    const cv::Point2f center = (pA + pB + pC + pD) / 4.f;
    const FLOAT_TYPE angle = static_cast<FLOAT_TYPE>(std::atan(lCD.M()) * 180.f / CV_PI);
    const cv::Size2f size(static_cast<float>(cv::norm(pC - pD)), static_cast<float>(cv::norm(pA - pD)));
    return cv::RotatedRect(center, size, angle);
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors(const cv::Mat &reference_image,
        std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE> &distances_line_cd, std::vector<cv::Point> &points,
        bool build_points) const
// -----------------------------------------------------------------------------------
{
    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    for (int y = valid_re.y; y < valid_re.y + valid_re.height; ++y)
    {
        const cv::Vec3b* ptr_row = reference_image.ptr<cv::Vec3b>(y);

        for (int x = valid_re.x; x < valid_re.x+valid_re.width; ++x)
        {
            const FLOAT_TYPE d_cd = std::abs(-lCD.M() * x + y - lCD.B()) / std::sqrt(1+lCD.M() * lCD.M());

            if (!hack_flash)
            {
                const FLOAT_TYPE d_ab = std::abs(-lAB.M() * x + y - lAB.B()) / std::sqrt(1 + lAB.M() * lAB.M());
                const FLOAT_TYPE d_bc = std::abs(-lBC.M() * x + y - lBC.B()) / std::sqrt(1 + lBC.M() * lBC.M());
                const FLOAT_TYPE d_da = std::abs(-lDA.M() * x + y - lDA.B()) / std::sqrt(1 + lDA.M() * lDA.M());

                if (std::abs((d_ab + d_bc + d_cd + d_da) - half_perimeter ) <= inner_rect_threshold)
                {
                    colors.push_back(cv::Vec3f(ptr_row[x][0], ptr_row[x][1], ptr_row[x][2]));
                    distances_line_cd.push_back(d_cd);

                    if (build_points)
                        points.push_back(cv::Point(x,y));
                }
            }
            else
            {
                colors.push_back(cv::Vec3f(ptr_row[x][0], ptr_row[x][1], ptr_row[x][2]));
                distances_line_cd.push_back(d_cd);

                if (build_points)
                    points.push_back( cv::Point(x,y) );
            }
        }
    }
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors(const cv::Mat &reference_image, const cv::Mat& valid_mask, const unsigned char mask_label,
        std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE> &distances_line_cd,
                                               std::vector<cv::Point> &points, bool build_points) const
// -----------------------------------------------------------------------------------
{
    assert(valid_mask.size()==reference_image.size());
    assert(valid_mask.type() == CV_8UC1);

    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    for (int y = valid_re.y; y < valid_re.y + valid_re.height; ++y)
    